"""MDS MCP server (FastMCP) exposing read-only NX-OS `show` commands as tools.

Connects to one or more Cisco MDS switches over NX-API.
"""

import os
import re
from typing import Any

import requests
import urllib3
from dotenv import load_dotenv
from fastmcp import FastMCP

# Generative UI lets the LLM build charts/tables/dashboards on the fly from
# `show` output, rendered inside MCP-Apps-capable clients.
from fastmcp.apps.generative import GenerativeUI

# Load variables from a local `.env` file if present (handy for local dev and
# HTTP mode). Real env vars (e.g. injected by an MCP client over stdio, or by
# the deployment platform's secret store) always take precedence and are never
# overwritten.
load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean from an environment variable.

    Treats "1", "true", "yes", "on" (case-insensitive) as True; anything else
    is False. Useful because env vars are always strings.
    """
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Switch registry — built once at startup from the indexed MDS_<N>_* env vars.
# We discover every index that defines a HOST and read its siblings (username,
# password, optional name/port/verify_ssl).
# ---------------------------------------------------------------------------


def _build_registry() -> dict[str, dict[str, Any]]:
    """Return {switch_name: {url, username, password, verify_ssl}}.

    Scans the environment for every `MDS_<N>_HOST` and builds one entry per
    index. Gaps in the numbering are tolerated (e.g. 1 and 3 without 2).

    Example return value::

        {
            "mds-fab-a": {
                "url": "https://192.0.2.10:8443/ins",
                "username": "admin",
                "password": "********",
                "verify_ssl": False,
            },
            "mds-fab-b": {
                "url": "https://192.0.2.11:8443/ins",
                "username": "admin",
                "password": "********",
                "verify_ssl": False,
            },
        }
    """
    registry: dict[str, dict[str, Any]] = {}

    indices = sorted(
        int(match.group(1))
        for key in os.environ
        if (match := re.fullmatch(r"MDS_(\d+)_HOST", key))
    )

    for i in indices:
        host = os.environ[f"MDS_{i}_HOST"].strip()
        username = os.getenv(f"MDS_{i}_USERNAME", "").strip()
        password = os.getenv(f"MDS_{i}_PASSWORD", "")
        if not (host and username and password):
            raise RuntimeError(
                f"Switch #{i} is missing one of "
                f"MDS_{i}_HOST / MDS_{i}_USERNAME / MDS_{i}_PASSWORD"
            )
        name = os.getenv(f"MDS_{i}_NAME", host).strip() or host
        port = os.getenv(f"MDS_{i}_PORT", "8443").strip()
        registry[name] = {
            "url": f"https://{host}:{port}/ins",
            "username": username,
            "password": password,
            "verify_ssl": _env_bool(f"MDS_{i}_VERIFY_SSL", default=False),
        }

    if not registry:
        raise RuntimeError(
            "No MDS switches configured. Set at least MDS_1_HOST / "
            "MDS_1_USERNAME / MDS_1_PASSWORD (see .env.example)."
        )

    return registry


SWITCHES = _build_registry()

# Silence the noisy InsecureRequestWarning if any switch disables TLS verification.
if any(not s["verify_ssl"] for s in SWITCHES.values()):
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ---------------------------------------------------------------------------
# Switch access — pick a switch and run a CLI NX-OS command against it.
# ---------------------------------------------------------------------------


def _resolve_switch(switch: str | None) -> tuple[str, dict[str, Any]]:
    """Return (name, config) for the requested switch.

    If `switch` is None and only one switch is registered, that switch is used.
    Otherwise raises a clear error listing the valid choices.
    """
    if switch is None:
        if len(SWITCHES) == 1:
            name = next(iter(SWITCHES))
            return name, SWITCHES[name]
        raise ValueError(
            f"Multiple MDS switches are configured ({sorted(SWITCHES)}). "
            "Pass the `switch` argument to choose one."
        )
    if switch not in SWITCHES:
        raise ValueError(
            f"Unknown switch '{switch}'. Valid choices: {sorted(SWITCHES)}."
        )
    return switch, SWITCHES[switch]


def _validate_command(cmd: str) -> str:
    """Allow only read-only `show` commands through NX-API.

    This server is intentionally read-only: every tool issues a `show ...`
    command. Enforcing it here is a defence-in-depth guard so no tool (or a
    future bug) can run a config / clear / reload command on the switch.
    Raises ValueError on anything that isn't a `show` command.
    """
    normalized = (cmd or "").strip()
    if not re.match(r"^show(\s|$)", normalized, re.IGNORECASE):
        raise ValueError(
            f"Refusing to run non-read-only command '{cmd}'. "
            "Only `show` commands are permitted."
        )
    return normalized


def _run_cli(switch: str | None, cmd: str) -> dict[str, Any]:
    """Run a single NX-OS CLI command on the chosen switch via NX-API (JSON-RPC).

    All tools below are thin wrappers around this helper — they only differ
    by the `cmd` string passed to the switch and the user-selected `switch`.
    """
    name, cfg = _resolve_switch(switch)
    cmd = _validate_command(cmd)

    # NX-API JSON-RPC payload: a list of one CLI command. `version: 1.2` is the
    # NX-API version.
    payload = [
        {
            "jsonrpc": "2.0",
            "method": "cli",
            "params": {"cmd": cmd, "version": 1.2},
            "id": 1,
        }
    ]

    # HTTP basic auth against the switch. `verify=...` lets us turn off TLS
    # verification for lab gear with self-signed certs.
    response = requests.post(
        cfg["url"],
        json=payload,
        headers={"content-type": "application/json-rpc"},
        auth=(cfg["username"], cfg["password"]),
        timeout=30,
        verify=cfg["verify_ssl"],
    )
    response.raise_for_status()
    data = response.json()
    # Wrap the response so the LLM always knows which switch produced it.
    return {"switch": name, "response": data}


# ---------------------------------------------------------------------------
# Validators — sanitise user/LLM-supplied tool arguments before they land
# inside an NX-API CLI string. These are the only command fragments not under
# our control, so each is validated strictly to prevent CLI injection.
# ---------------------------------------------------------------------------

# Zone names on MDS: start with a letter, then letters/digits/`_ $ ^ -`,
# up to 64 chars. Used by `show_zone_name`.
_ZONE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_$^-]{0,63}$")


def _validate_zone_name(zone: str) -> str:
    """Reject anything that isn't a syntactically valid MDS zone name.

    Accepts a name beginning with a letter followed by letters, digits, or
    `_ $ ^ -` (max 64 chars). Raises ValueError otherwise.
    """
    name = (zone or "").strip()
    if not _ZONE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid zone name '{zone}'. Zone names must start with a letter "
            "and contain only letters, digits, and the characters _ $ ^ - "
            "(max 64 characters)."
        )
    return name


def _validate_zoneset_name(zoneset: str) -> str:
    """Reject anything that isn't a syntactically valid MDS zoneset name.

    Zoneset names follow the same rules as zone names: begin with a letter,
    then letters, digits, or `_ $ ^ -` (max 64 chars). Raises ValueError
    otherwise.
    """
    name = (zoneset or "").strip()
    if not _ZONE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid zoneset name '{zoneset}'. Zoneset names must start with a "
            "letter and contain only letters, digits, and the characters "
            "_ $ ^ - (max 64 characters)."
        )
    return name


# Device-alias names on MDS: start with a letter, then letters/digits/`_ -`,
# up to 64 chars. Used by `show_device_alias_name`.
_DEVICE_ALIAS_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def _validate_device_alias_name(name: str) -> str:
    """Reject anything that isn't a syntactically valid MDS device-alias name.

    Accepts a name beginning with a letter followed by letters, digits, or
    `_ -` (max 64 chars). Raises ValueError otherwise.
    """
    alias = (name or "").strip()
    if not _DEVICE_ALIAS_NAME_RE.match(alias):
        raise ValueError(
            f"Invalid device-alias name '{name}'. Device-alias names must start "
            "with a letter and contain only letters, digits, and the characters "
            "_ - (max 64 characters)."
        )
    return alias


# Interface names on MDS: `fcX/Y`, `fcX/Y/Z` (fabric-extender ports) or
# `port-channelN`. Used by the per-interface tools.
_INTERFACE_RE = re.compile(r"^(fc\d+/\d+(?:/\d+)?|port-channel\d+)$")


def _validate_interface(interface: str) -> str:
    """Reject anything that doesn't look like a real MDS interface name.

    Accepts `fcX/Y`, `fcX/Y/Z` (for fabric-extender ports) and `port-channelN`.
    Tolerates common LLM/user formatting quirks: extra whitespace, a space
    between the interface kind and its number (e.g. `port-channel 1`,
    `fc 1/1`), and common synonyms (`po1`, `Po1`, `portchannel1`).
    Raises ValueError on anything else.
    """
    raw = (interface or "").strip().lower()
    # Collapse internal whitespace, e.g. "port-channel 1" -> "port-channel1",
    # "fc 1 / 1" -> "fc1/1".
    name = re.sub(r"\s+", "", raw)
    # Accept common port-channel synonyms.
    name = re.sub(r"^(po|portchannel)(\d+)$", r"port-channel\2", name)
    if not _INTERFACE_RE.match(name):
        raise ValueError(
            f"Invalid interface name '{interface}'. Expected e.g. 'fc1/1' "
            "or 'port-channel1'."
        )
    return name


def _validate_vsan(vsan: int) -> int:
    """Reject anything that isn't a valid MDS VSAN id (1-4094).

    Accepts an int (or an int-like string); the resulting digits are the only
    user-controlled fragment that lands inside the CLI string. Raises
    ValueError on a non-integer or out-of-range value.
    """
    try:
        vsan_id = int(vsan)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid VSAN '{vsan}'. Expected an integer 1-4094."
        ) from None
    if not 1 <= vsan_id <= 4094:
        raise ValueError(f"Invalid VSAN '{vsan}'. VSAN id must be between 1 and 4094.")
    return vsan_id


# pWWN: 8 colon-separated hex bytes, e.g. `50:00:00:00:00:00:00:01`. Used by
# `show_device_alias_pwwn`.
_PWWN_RE = re.compile(r"^(?:[0-9A-Fa-f]{2}:){7}[0-9A-Fa-f]{2}$")


def _validate_pwwn(pwwn: str) -> str:
    """Reject anything that isn't a valid port WWN (8 colon-separated hex bytes).

    Accepts e.g. `50:00:00:00:00:00:00:01`; normalises to lowercase. The hex
    digits are the only user-controlled fragment that lands inside the CLI
    string. Raises ValueError on anything malformed.
    """
    wwn = (pwwn or "").strip().lower()
    if not _PWWN_RE.match(wwn):
        raise ValueError(
            f"Invalid pWWN '{pwwn}'. Expected 8 colon-separated hex bytes, "
            "e.g. '50:00:00:00:00:00:00:01'."
        )
    return wwn


# ---------------------------------------------------------------------------
# MCP server instance — "cisco-mds" is the server name advertised to MCP clients
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "cisco-mds",
    instructions="Read-only Cisco MDS switch tools. Only `show` commands are permitted.",
)

# Generative UI: registers `generate_prefab_ui` (the LLM writes Prefab code that
# is rendered live in MCP-Apps-capable clients) and `search_prefab_components`.
# Lets the model build charts/tables/dashboards on the fly from `show` output.
mcp.add_provider(GenerativeUI())


# ===========================================================================
# ||                                                                       ||
# ||                            MDS MCP TOOLS                              ||
# ||                                                                       ||
# ===========================================================================


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@mcp.tool
def list_switches() -> dict[str, Any]:
    """List the MDS switches this server can talk to.

    Call this FIRST whenever the operator hasn't named a switch, OR before
    passing a `switch` argument whose exact spelling you are not certain of.
    Switch names are CASE-SENSITIVE and must match exactly: `mds-fab-a` and
    `MDS-FAB-A` are treated as two different switches. Do not guess or
    normalise the casing the operator typed — instead, look up the closest
    matching name returned here and pass that exact string to other tools.
    Every other tool accepts a `switch` argument that must be one of the
    names returned here.

    Returns a dict with:
      - `switches`: list of {`name`, `url`} entries (credentials are NOT returned).
      - `default` (str | None): the name used when a tool is invoked without
        `switch`; set only when exactly one switch is configured, otherwise None.

    Example::

        {
            "switches": [{"name": "mds-fab-a", "url": "https://192.0.2.10:8443/ins"}],
            "default": "mds-fab-a",
        }
    """
    return {
        "switches": [{"name": n, "url": c["url"]} for n, c in SWITCHES.items()],
        "default": next(iter(SWITCHES)) if len(SWITCHES) == 1 else None,
    }


# ---------------------------------------------------------------------------
# Zoning
# ---------------------------------------------------------------------------

# Zones


@mcp.tool
def show_zone(switch: str | None = None) -> dict[str, Any]:
    """Return ALL zones configured on the MDS switch (`show zone`).

    Dumps the complete zone database: every zone in every VSAN, both active
    (part of the currently enforced zoneset) and inactive (configured but not
    activated). Use this when the operator asks about FC zoning, zone
    membership, which initiators can talk to which targets, or to audit zone
    hygiene. For only the zones currently being enforced, see `show_zoneset`
    (active zoneset).

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The zones are at
    `response.result.body.TABLE_zone.ROW_zone` — a flat list (NOT grouped by
    VSAN), where each zone has:
      - `name` (str) — the zone name
      - `vsan` (int) — the VSAN this zone belongs to
      - `TABLE_zone_member.ROW_zone_member` — list of members, each with:
          * `type` — either `"pwwn"` or `"device-alias"`
          * for `"pwwn"`: `wwn` (the pWWN), plus `dev_alias` when that WWN
            resolves to a device-alias
          * for `"device-alias"`: `dev_alias` (the name) and `dev_alias_pwwn`
            (the WWN it maps to)
    A single zone commonly mixes both member types. Note the NX-API quirk:
    `ROW_zone` and `ROW_zone_member` are returned as a single object instead
    of a list when there is exactly one entry.
    """
    return _run_cli(switch, "show zone")


@mcp.tool
def show_zone_active(switch: str | None = None) -> dict[str, Any]:
    """Return only the ACTIVE zones being enforced (`show zone active`).

    Unlike `show_zone` (which dumps the full configured zone database,
    including inactive zones), this returns just the zones that belong to the
    currently activated zoneset and are therefore actively enforcing traffic
    isolation in the fabric. Use this to answer "which devices can actually
    talk to each other right now?". A key extra field here is the live FC-ID
    (`online_fcid`) for members that are currently logged in.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The active zones are at
    `response.result.body.TABLE_zone_active.ROW_zone_active` — a flat list
    (NOT grouped by VSAN), where each zone has:
      - `name` (str) — the zone name
      - `vsan` (int) — the VSAN this zone belongs to
      - `TABLE_zone_member.ROW_zone_member` — list of members, each with:
          * `type` — either `"pwwn"` or `"device-alias"`
          * `online_fcid` (str) — the device's live FC-ID when currently
            logged in, or `""` when offline / not logged in
          * for `"pwwn"`: `wwn` (the pWWN), plus `dev_alias` when that WWN
            resolves to a device-alias
          * for `"device-alias"`: `dev_alias` (the name) and `dev_alias_pwwn`
            (the WWN it maps to)
    A single zone commonly mixes both member types. Note the NX-API quirk:
    `ROW_zone_active` and `ROW_zone_member` are returned as a single object
    instead of a list when there is exactly one entry.
    """
    return _run_cli(switch, "show zone active")


@mcp.tool
def show_zone_name(zone: str, switch: str | None = None) -> dict[str, Any]:
    """Return a single named zone and its members (`show zone name <zone>`).

    Use this when the operator already knows the zone name and wants just
    that one zone's membership, instead of dumping the whole database with
    `show_zone`. Zone names are CASE-SENSITIVE and must match exactly; if you
    are unsure of the spelling, call `show_zone` first to discover the exact
    name.

    Args:
      zone: Exact, case-sensitive zone name (e.g. `zone-01`). Validated
        against `^[A-Za-z][A-Za-z0-9_$^-]{0,63}$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The zone is at
    `response.result.body.TABLE_zone.ROW_zone` (a list, normally with a single
    entry for the requested name), where the zone has:
      - `name` (str) — the zone name
      - `vsan` (int) — the VSAN this zone belongs to
      - `TABLE_zone_member.ROW_zone_member` — list of members, each with:
          * `type` — either `"pwwn"` or `"device-alias"`
          * for `"pwwn"`: `wwn` (the pWWN), plus `dev_alias` when that WWN
            resolves to a device-alias
          * for `"device-alias"`: `dev_alias` (the name) and `dev_alias_pwwn`
            (the WWN it maps to)
    Note the NX-API quirk: `ROW_zone` and `ROW_zone_member` are returned as a
    single object instead of a list when there is exactly one entry.
    """
    name = _validate_zone_name(zone)
    return _run_cli(switch, f"show zone name {name}")


@mcp.tool
def show_zone_vsan(vsan: int, switch: str | None = None) -> dict[str, Any]:
    """Return all zones in a single VSAN (`show zone vsan <vsan>`).

    Use this to scope the zone database to one VSAN instead of dumping every
    VSAN with `show_zone`. Handy when the operator is troubleshooting a
    specific fabric/VSAN (e.g. "what zones exist in VSAN 20?").

    Args:
      vsan: VSAN id to query (integer 1-4094, e.g. `20`).
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The zones are at
    `response.result.body.TABLE_zone.ROW_zone` — a list (all zones share the
    requested `vsan`), where each zone has:
      - `name` (str) — the zone name
      - `vsan` (int) — the VSAN this zone belongs to (matches the argument)
      - `TABLE_zone_member.ROW_zone_member` — list of members, each with:
          * `type` — either `"pwwn"` or `"device-alias"`
          * for `"pwwn"`: `wwn` (the pWWN), plus `dev_alias` when that WWN
            resolves to a device-alias
          * for `"device-alias"`: `dev_alias` (the name) and `dev_alias_pwwn`
            (the WWN it maps to)
    Note the NX-API quirk: `ROW_zone` and `ROW_zone_member` are returned as a
    single object instead of a list when there is exactly one entry.
    """
    vsan_id = _validate_vsan(vsan)
    return _run_cli(switch, f"show zone vsan {vsan_id}")


@mcp.tool
def show_zone_name_vsan(
    zone: str, vsan: int, switch: str | None = None
) -> dict[str, Any]:
    """Return one named zone within a specific VSAN
    (`show zone name <zone> vsan <vsan>`).

    Use this to disambiguate when the same zone name exists in more than one
    VSAN, or simply to fetch a single zone scoped to its VSAN. More precise
    than `show_zone_name` (which is not VSAN-scoped).

    Args:
      zone: Exact, case-sensitive zone name (e.g. `esx-host-01`). Validated
        against `^[A-Za-z][A-Za-z0-9_$^-]{0,63}$`.
      vsan: VSAN id the zone lives in (integer 1-4094, e.g. `20`).
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Unlike `show_zone` / `show_zone_name`,
    the zone object sits DIRECTLY at `response.result.body` (there is no
    `TABLE_zone` / `ROW_zone` wrapper since exactly one zone is returned):
      - `name` (str) — the zone name
      - `vsan` (int) — the VSAN this zone belongs to (matches the argument)
      - `TABLE_zone_member.ROW_zone_member` — list of members, each with:
          * `type` — either `"pwwn"` or `"device-alias"`
          * for `"pwwn"`: `wwn` (the pWWN), plus `dev_alias` when that WWN
            resolves to a device-alias
          * for `"device-alias"`: `dev_alias` (the name) and `dev_alias_pwwn`
            (the WWN it maps to)
    Note the NX-API quirk: `ROW_zone_member` is returned as a single object
    instead of a list when the zone has exactly one member.
    """
    name = _validate_zone_name(zone)
    vsan_id = _validate_vsan(vsan)
    return _run_cli(switch, f"show zone name {name} vsan {vsan_id}")


@mcp.tool
def show_zone_status(switch: str | None = None) -> dict[str, Any]:
    """Return zone server status & last activation result (`show zone status`).

    Use this when a zoneset activation failed, or to audit zoning policy
    settings (default-zone behaviour, smart-zoning, distribution). It also
    reveals which zoneset is active per VSAN (`activedb_zoneset_name`).

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). One entry per VSAN at
    `response.result.body.TABLE_zone_status.ROW_zone_status`, with fields:
      - `vsan` (int)
      - Policy: `mode` (`basic` / `enhanced`), `default_zone`
        (`deny` / `permit` — `permit` is unsafe in production), `distribute`
        (e.g. `active only` / `full`), `interop`, `merge_control`
        (`allow` / `restrict`), `smart_zoning` (`enabled` / `disabled`),
        `hard_zoning`, `rscn_format`, `activation_overwrite_control`
      - `session` — current zone-server session state (`none` when idle)
      - Active zoneset (present only when one is activated in the VSAN):
          * `activedb_zoneset_name` (str) — the active zoneset's name
          * `activedb_zoneset_count`, `activedb_zone_count`, `activedb_dbsize`
      - Full (configured) DB: `fulldb_zoneset_count`, `fulldb_zone_count`,
        `fulldb_aliases`, `fulldb_dbsize`
      - Sizing: `effectivedb_dbsize`, `maxdb_dbsize`, `percent_effectivedbsize`,
        `sfcsize`, `max_sfcsize`, `percent_sfcsize`
      - `status` (str) — last activation result text, e.g.
        "Activation completed at 00:00:00 UTC Jan 01 2025" (empty when the
        VSAN has never had a zoneset activated)
    """
    return _run_cli(switch, "show zone status")


# Zonesets


@mcp.tool
def show_zoneset(switch: str | None = None) -> dict[str, Any]:
    """Return every configured zoneset and its member zones (`show zoneset`).

    A zoneset is a named group of zones; only ONE zoneset can be active per
    VSAN at any time. Use this to see the zonesets defined on the switch and,
    crucially, the zones (and their members) each one contains. This lists
    EVERY configured zoneset, and its JSON has no flag marking which one is
    currently activated — to find the active policy use `show_zoneset_active`
    (returns only the active zoneset) or `show_zone_status` (reports the
    per-VSAN activation state).

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). This payload is nested three levels
    deep. The zonesets are at
    `response.result.body.TABLE_zoneset.ROW_zoneset` — a list, where each
    zoneset has:
      - `name` (str) — the zoneset name
      - `vsan` (int) — the VSAN it belongs to
      - `TABLE_zone.ROW_zone` — list of the zones in this zoneset, each with:
          * `name` (str) — the zone name
          * `vsan` (int)
          * `TABLE_member.ROW_member` — list of members (NOTE: the key here is
            `TABLE_member`/`ROW_member`, NOT `TABLE_zone_member` as in
            `show_zone`), each with:
              - `type` — either `"pwwn"` or `"device-alias"`
              - `online_fcid` (str) — live FC-ID when logged in, else `""`
              - for `"pwwn"`: `wwn`, plus `dev_alias` when it resolves
              - for `"device-alias"`: `dev_alias` and `dev_alias_pwwn`
    Note the NX-API quirk: every `ROW_*` table collapses to a single object
    instead of a list when it has exactly one entry.
    """
    return _run_cli(switch, "show zoneset")


@mcp.tool
def show_zoneset_active(switch: str | None = None) -> dict[str, Any]:
    """Return only the ACTIVE zoneset and its zones (`show zoneset active`).

    Unlike `show_zoneset` (which lists every configured zoneset), this returns
    just the zoneset currently activated per VSAN — the policy actually being
    enforced. Members carry a live `online_fcid` so you can also see which
    devices are logged in. Use this to answer "what zoning is in effect right
    now?".

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Same three-level structure as
    `show_zoneset`, at `response.result.body.TABLE_zoneset.ROW_zoneset` — a
    list (normally one active zoneset per VSAN), where each zoneset has:
      - `name` (str) — the active zoneset name
      - `vsan` (int) — the VSAN it belongs to
      - `TABLE_zone.ROW_zone` — list of the zones in this zoneset, each with:
          * `name` (str) — the zone name
          * `vsan` (int)
          * `TABLE_member.ROW_member` — list of members (NOTE: the key is
            `TABLE_member`/`ROW_member`, NOT `TABLE_zone_member`), each with:
              - `type` — either `"pwwn"` or `"device-alias"`
              - `online_fcid` (str) — the device's live FC-ID when currently
                logged in, or `""` when offline / not logged in
              - for `"pwwn"`: `wwn`, plus `dev_alias` when it resolves
              - for `"device-alias"`: `dev_alias` and `dev_alias_pwwn`
    Note the NX-API quirk: every `ROW_*` table collapses to a single object
    instead of a list when it has exactly one entry.
    """
    return _run_cli(switch, "show zoneset active")


@mcp.tool
def show_zoneset_active_vsan(vsan: int, switch: str | None = None) -> dict[str, Any]:
    """Return the ACTIVE zoneset in one VSAN (`show zoneset active vsan <id>`).

    Like `show_zoneset_active` but scoped to a single VSAN, so the response
    only carries the policy enforced in that fabric instead of the active
    zoneset of every VSAN. Use this to answer "what zoning is in effect right
    now in VSAN <id>?".

    Args:
      vsan: VSAN id to query (1-4094). Validated before being sent.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Same three-level structure as
    `show_zoneset_active`, at `response.result.body.TABLE_zoneset.ROW_zoneset`
    — a list (normally the one active zoneset in the VSAN), where each zoneset
    has:
      - `name` (str) — the active zoneset name
      - `vsan` (int) — the VSAN it belongs to
      - `TABLE_zone.ROW_zone` — list of the zones, each with:
          * `name` (str), `vsan` (int)
          * `TABLE_member.ROW_member` — list of members (key is
            `TABLE_member`/`ROW_member`, NOT `TABLE_zone_member`), each with:
              - `type` — either `"pwwn"` or `"device-alias"`
              - `online_fcid` (str) — live FC-ID when logged in, else `""`
              - for `"pwwn"`: `wwn`, plus `dev_alias` when it resolves
              - for `"device-alias"`: `dev_alias` and `dev_alias_pwwn`
    Note the NX-API quirk: every `ROW_*` table collapses to a single object
    instead of a list when it has exactly one entry.
    """
    vsan_id = _validate_vsan(vsan)
    return _run_cli(switch, f"show zoneset active vsan {vsan_id}")


@mcp.tool
def show_zoneset_name(zoneset: str, switch: str | None = None) -> dict[str, Any]:
    """Return one named zoneset and its zones (`show zoneset name <name>`).

    Like `show_zoneset` but scoped to a single zoneset, so the response is not
    cluttered with the other zonesets configured on the switch. Use this when
    you already know the zoneset name (e.g. from `show_zone_status`'s
    `activedb_zoneset_name`) and just want its zones and members.

    Args:
      zoneset: Name of the zoneset to query. Validated against the MDS
        naming rules before being sent to the device.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Same three-level structure as
    `show_zoneset`, at `response.result.body.TABLE_zoneset.ROW_zoneset` — a
    list (one entry for the matched zoneset), where each zoneset has:
      - `name` (str) — the zoneset name
      - `vsan` (int) — the VSAN it belongs to
      - `TABLE_zone.ROW_zone` — list of the zones, each with:
          * `name` (str), `vsan` (int)
          * `TABLE_member.ROW_member` — list of members (key is
            `TABLE_member`/`ROW_member`, NOT `TABLE_zone_member`), each with:
              - `type` — either `"pwwn"` or `"device-alias"`
              - `online_fcid` (str) — live FC-ID when logged in, else `""`
              - for `"pwwn"`: `wwn`, plus `dev_alias` when it resolves
              - for `"device-alias"`: `dev_alias` and `dev_alias_pwwn`
    Note the NX-API quirk: every `ROW_*` table collapses to a single object
    instead of a list when it has exactly one entry.
    """
    name = _validate_zoneset_name(zoneset)
    return _run_cli(switch, f"show zoneset name {name}")


@mcp.tool
def show_zoneset_vsan(vsan: int, switch: str | None = None) -> dict[str, Any]:
    """Return the zonesets configured in one VSAN (`show zoneset vsan <id>`).

    Like `show_zoneset` but restricted to a single VSAN, which is the common
    case since zonesets are per-VSAN. Use this to inspect the zonesets (and
    their zones/members) defined in a specific fabric.

    Args:
      vsan: VSAN id to query (1-4094). Validated before being sent.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Same three-level structure as
    `show_zoneset`, at `response.result.body.TABLE_zoneset.ROW_zoneset` — a
    list of the zonesets in the VSAN, each with:
      - `name` (str) — the zoneset name
      - `vsan` (int) — the VSAN it belongs to
      - `TABLE_zone.ROW_zone` — list of the zones, each with:
          * `name` (str), `vsan` (int)
          * `TABLE_member.ROW_member` — list of members (key is
            `TABLE_member`/`ROW_member`, NOT `TABLE_zone_member`), each with:
              - `type` — either `"pwwn"` or `"device-alias"`
              - `online_fcid` (str) — live FC-ID when logged in, else `""`
              - for `"pwwn"`: `wwn`, plus `dev_alias` when it resolves
              - for `"device-alias"`: `dev_alias` and `dev_alias_pwwn`
    Note the NX-API quirk: every `ROW_*` table collapses to a single object
    instead of a list when it has exactly one entry.
    """
    vsan_id = _validate_vsan(vsan)
    return _run_cli(switch, f"show zoneset vsan {vsan_id}")


@mcp.tool
def show_zoneset_name_vsan(
    zoneset: str, vsan: int, switch: str | None = None
) -> dict[str, Any]:
    """Return one named zoneset in one VSAN (`show zoneset name <name> vsan <id>`).

    The most specific zoneset lookup: combines `show_zoneset_name` and
    `show_zoneset_vsan` to fetch exactly one zoneset in exactly one VSAN. Use
    this when the same zoneset name could exist in multiple VSANs and you want
    to disambiguate.

    Args:
      zoneset: Name of the zoneset to query. Validated against the MDS
        naming rules before being sent.
      vsan: VSAN id to query (1-4094). Validated before being sent.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Same three-level structure as
    `show_zoneset`, at `response.result.body.TABLE_zoneset.ROW_zoneset` — a
    list (one entry for the matched zoneset), where each zoneset has:
      - `name` (str) — the zoneset name
      - `vsan` (int) — the VSAN it belongs to
      - `TABLE_zone.ROW_zone` — list of the zones, each with:
          * `name` (str), `vsan` (int)
          * `TABLE_member.ROW_member` — list of members (key is
            `TABLE_member`/`ROW_member`, NOT `TABLE_zone_member`), each with:
              - `type` — either `"pwwn"` or `"device-alias"`
              - `online_fcid` (str) — live FC-ID when logged in, else `""`
              - for `"pwwn"`: `wwn`, plus `dev_alias` when it resolves
              - for `"device-alias"`: `dev_alias` and `dev_alias_pwwn`
    Note the NX-API quirk: every `ROW_*` table collapses to a single object
    instead of a list when it has exactly one entry.
    """
    name = _validate_zoneset_name(zoneset)
    vsan_id = _validate_vsan(vsan)
    return _run_cli(switch, f"show zoneset name {name} vsan {vsan_id}")


# ---------------------------------------------------------------------------
# Devices Aliases
# ---------------------------------------------------------------------------


@mcp.tool
def show_device_alias_database(switch: str | None = None) -> dict[str, Any]:
    """Return the device-alias database (`show device-alias database`).

    Use this to translate pWWNs (e.g. `50:00:00:00:00:00:00:01`)
    into human-friendly names like `esx-host-01-hba-a`, or vice-versa.
    Device-aliases are fabric-wide (not VSAN-scoped), unlike fcaliases, and
    are the names referenced by zone members (see `show_zone` /
    `show_zoneset`). Good first call when correlating zoning, FLOGI or FCNS
    output back to recognisable host/array names.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The entries are at
    `response.result.body.TABLE_device_alias_database.ROW_device_alias_database`
    — a list, where each entry has:
      - `dev_alias_name` (str) — the friendly name
      - `pwwn` (str) — the 8-byte port WWN it maps to
    Alongside the table, `response.result.body.number_of_entries` (int) gives
    the total alias count. Note the NX-API quirk: `ROW_device_alias_database`
    collapses to a single object instead of a list when there is exactly one
    alias defined.
    """
    return _run_cli(switch, "show device-alias database")


@mcp.tool
def show_device_alias_name(name: str, switch: str | None = None) -> dict[str, Any]:
    """Return a single device-alias mapping (`show device-alias name <name>`).

    Use this when you already know the alias and just want its pWWN, instead
    of dumping the whole database with `show_device_alias_database`. Handy for
    confirming what a zone member like `host-01-hba-a` actually resolves to.

    Args:
      name: Exact device-alias name (e.g. `host-01-hba-a`). Validated against
        the MDS naming rules before being sent to the device.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Unlike the database dump, there is no
    TABLE/ROW wrapper — the single entry sits DIRECTLY at
    `response.result.body`:
      - `dev_alias_name` (str) — the friendly name
      - `pwwn` (str) — the 8-byte port WWN it maps to
    """
    alias = _validate_device_alias_name(name)
    return _run_cli(switch, f"show device-alias name {alias}")


@mcp.tool
def show_device_alias_pwwn(pwwn: str, switch: str | None = None) -> dict[str, Any]:
    """Return the device-alias for a pWWN (`show device-alias pwwn <pwwn>`).

    The reverse of `show_device_alias_name`: given a port WWN, look up the
    friendly alias it maps to. Use this to put a name to a raw pWWN seen in
    FLOGI/FCNS output or in a zone member, without dumping the whole database.

    Args:
      pwwn: Port WWN as 8 colon-separated hex bytes (e.g.
        `50:00:00:00:00:00:00:01`). Validated before being sent to the device.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Like `show_device_alias_name`, there
    is no TABLE/ROW wrapper — the single entry sits DIRECTLY at
    `response.result.body`:
      - `dev_alias_name` (str) — the friendly name
      - `pwwn` (str) — the 8-byte port WWN it maps to
    """
    wwn = _validate_pwwn(pwwn)
    return _run_cli(switch, f"show device-alias pwwn {wwn}")


# ---------------------------------------------------------------------------
# System / Inventory
# ---------------------------------------------------------------------------


@mcp.tool
def show_hardware(switch: str | None = None) -> dict[str, Any]:
    """Return hardware platform & boot details (`show hardware`).

    Use this for questions about the chassis, supervisor / linecard modules,
    power supplies, fans, bootflash contents, NX-OS version banner, uptime,
    or "what kind of switch am I talking to?".

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Scalar fields under
    `response.result.body` include:
      - `chassis_id` (str, e.g. "MDS 9132T 32X32G FC (1 RU) Chassis")
      - `host_name`, `manufacturer`, `proc_board_id` (chassis serial)
      - Software: `bios_ver_str`, `kickstart_ver_str`, `sys_ver_str`,
        `kick_file_name` / `isan_file_name` (boot images) and their compile /
        timestamp fields
      - `cpu_name`, `memory` (+ `mem_type`), `bootflash_size`
      - Uptime: `kern_uptm_days` / `kern_uptm_hrs` / `kern_uptm_mins` /
        `kern_uptm_secs`
      - Last reload: `rr_reason`, `rr_sys_ver`, `rr_service`
    Hardware FRUs are under `body.TABLE_slot_info` — a list, one entry per
    slot category (chassis, modules, power supplies, fans). Each has a
    `ROW_slot_info` list whose rows carry `type`, `model_num`, `serial_num`,
    `hw_ver`, `part_num`, `CLEI_code`, and a `status_ok_empty` string (e.g.
    "PS1 ok", "Fan1 ok"). For multi-component slots several of these fields
    are themselves LISTS (parallel arrays), and `body.num_slot_str` summarises
    the slot counts. For a flat FRU list with serials use `show_inventory`.
    """
    return _run_cli(switch, "show hardware")


@mcp.tool
def show_version(switch: str | None = None) -> dict[str, Any]:
    """Return software, hardware and uptime info (`show version`).

    Use this to report the NX-OS version, the reason for the last reboot,
    chassis identity, and bootflash images currently loaded. Lighter than
    `show_hardware` and ideal for "what version is this switch running?".

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Common scalar fields under
    `response.result.body` (same shape as `show_hardware` but with NO
    `TABLE_slot_info`):
      - Software: `kickstart_ver_str`, `sys_ver_str` (running version),
        `bios_ver_str`, `loader_ver_str`, plus boot images `kick_file_name` /
        `isan_file_name` and their compile/timestamp fields
      - Identity: `chassis_id`, `module_id`, `host_name`, `manufacturer`,
        `proc_board_id` (serial)
      - Resources: `cpu_name`, `memory` (+ `mem_type`), `bootflash_size`
      - Uptime: `kern_uptm_days` / `kern_uptm_hrs` / `kern_uptm_mins` /
        `kern_uptm_secs` (since last reboot)
      - Last reload: `rr_reason`, `rr_sys_ver`, `rr_service`
      - `header_str` — the NX-OS copyright/banner block
    """
    return _run_cli(switch, "show version")


@mcp.tool
def show_inventory(switch: str | None = None) -> dict[str, Any]:
    """Return the full FRU inventory (`show inventory`).

    Use this for asset management, RMA preparation, or to enumerate every
    field-replaceable unit (chassis, supervisor, expansion modules, fan
    modules, power supplies) with vendor PIDs and serial numbers.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The FRUs are at
    `response.result.body.TABLE_inv.ROW_inv` — a list, where each entry has:
      - `name` (str) — slot/bay identifier (e.g. `Chassis`, `Slot 1`,
        `Slot 4`); note multiple rows can share a slot (e.g. two `Slot 1`
        entries for a SUP plus its expansion module)
      - `desc` (str) — human-readable description
      - `productid` (str) — Cisco PID
      - `vendorid` (str)
      - `serialnum` (str) — serial used for RMA/warranty, or `"N/A"` for FRUs
        without one (e.g. fan modules)
    Note the NX-API quirk: `ROW_inv` collapses to a single object instead of a
    list when there is exactly one FRU.
    """
    return _run_cli(switch, "show inventory")


@mcp.tool
def show_module(switch: str | None = None) -> dict[str, Any]:
    """Return per-slot module status (`show module`).

    Use this to verify supervisor and linecard health, diagnose modules in
    `failure` / `pwr-denied` / `unknown` state, or confirm an HA standby is
    truly `ha-standby` before doing maintenance.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). `response.result.body` holds three
    parallel `TABLE_*` / `ROW_*` lists keyed by `mod` (slot number):
      - `TABLE_modinfo.ROW_modinfo` — `mod`, `ports` (int), `modtype` (e.g.
        "16X4/8/16/32G FC Sup + 16X32G FC LEM"), `model`, `status` (e.g.
        `active *`, `ha-standby`, `ok`, `powered-dn`, `pwr-denied`,
        `failure`); the `*` marks the active supervisor
      - `TABLE_modwwninfo.ROW_modwwninfo` — `mod`, `sw` (running NX-OS), `hw`
        (hardware revision), `wwn` (the slot's WWN range)
      - `TABLE_modmacinfo.ROW_modmacinfo` — `mod`, `mac` (MAC range),
        `serialnum`
    Note the NX-API quirk: each `ROW_*` collapses to a single object instead
    of a list when there is exactly one module.
    """
    return _run_cli(switch, "show module")


@mcp.tool
def show_module_uptime(switch: str | None = None) -> dict[str, Any]:
    """Return how long each module has been up (`show module uptime`).

    Use this to spot a module (typically the supervisor) that rebooted more
    recently than the others — handy when correlating an outage to an
    unexpected module restart, or confirming uptime before maintenance.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The per-module uptime rows are at
    `response.result.body.TABLE_uptimeinf.ROW_uptimeinf` — a list, where each
    entry has:
      - `starttime` (str) — when the module last came up, e.g.
        "Mon Jan 01 00:00:00 2025"
      - `daysup`, `hoursup`, `minutesup`, `secondsup` (int) — elapsed uptime
    Note the NX-API quirk: `ROW_uptimeinf` collapses to a single object
    instead of a list when there is exactly one module.
    """
    return _run_cli(switch, "show module uptime")


@mcp.tool
def show_boot(switch: str | None = None) -> dict[str, Any]:
    """Return the configured boot images (`show boot`).

    Use this to verify which kickstart/system images the switch is running now
    versus what it will load on the next reload — essential before/after an
    upgrade to confirm the startup boot variables were saved. Complements
    `show_version` (which reports the currently-loaded images and last reload
    reason).

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). `response.result.body` holds two
    parallel tables — the running (current) and saved (startup) boot config:
      - `TABLE_Current_Bootvar.ROW_Current_Bootvar` — currently-active boot
        variables, each entry has `current_sup_module` (slot, may be empty for
        a fixed/single-sup chassis), `current_kickstart` (kickstart image
        path, e.g. `bootflash:/...-kickstart-mz.<ver>.bin`), `current_system`
        (system image path), `current_poap_status` (e.g. `Disabled`),
        `current_boot_var` (e.g. `No module boot variable set`)
      - `TABLE_Startup_Bootvar.ROW_Startup_Bootvar` — boot variables saved in
        startup-config (loaded on next reload), with the same fields prefixed
        `start_` instead of `current_`: `start_sup_module`, `start_kickstart`,
        `start_system`, `start_poap_status`, `start_boot_var`
    A mismatch between the Current and Startup images means a pending change
    has not been saved (run `copy running-config startup-config`). Note the
    NX-API quirk: each `ROW_*` collapses to a single object instead of a list
    when there is exactly one entry.
    """
    return _run_cli(switch, "show boot")


@mcp.tool
def show_switchname(switch: str | None = None) -> dict[str, Any]:
    """Return the switch's configured hostname / FQDN (`show switchname`).

    Lightweight way to confirm which physical device you reached — useful to
    cross-check that the registry `switch` name maps to the real hostname.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). NOTE: this command has no structured
    body — the hostname is a plain string at `response.result.msg` (e.g.
    "mds-fab-a.example.com\\n"), with a trailing newline.
    """
    return _run_cli(switch, "show switchname")


@mcp.tool
def show_cdp_neighbors(switch: str | None = None) -> dict[str, Any]:
    """Return directly-connected CDP neighbors (`show cdp neighbors`).

    Use this to map physical topology — discover which upstream/peer Cisco
    devices are cabled to this switch and on which local/remote ports. Handy
    for verifying mgmt0 uplinks and ISL cabling, or confirming a neighbor's
    platform before maintenance. (CDP only discovers Cisco devices.)

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The neighbor rows are at
    `response.result.body.TABLE_cdp_neighbor_brief_info.ROW_cdp_neighbor_brief_info`
    — a list, where each entry has:
      - `device_id` (str) — neighbor hostname, usually with its serial in
        parentheses (e.g. `peer-sw-01.example.com(SERIAL123)`)
      - `interface` (str) — the LOCAL port on this switch (e.g. `mgmt0`)
      - `ttl` (int) — seconds until this CDP entry expires
      - `capability` (list[str]) — neighbor roles (e.g. `router`, `switch`,
        `IGMP_cnd_filtering`, `Supports-STP-Dispute`)
      - `platform_id` (str) — neighbor hardware model (e.g. `N9K-C93180YC-FX`)
      - `port_id` (str) — the REMOTE port on the neighbor (e.g. `Ethernet1/1`)
    `response.result.body.neigh_count` (int) gives the total neighbor count.
    Note the NX-API quirks: `ROW_cdp_neighbor_brief_info` collapses to a single
    object instead of a list when there is exactly one neighbor, and
    `capability` collapses to a bare string when there is only one capability.
    """
    return _run_cli(switch, "show cdp neighbors")


@mcp.tool
def show_hosts(switch: str | None = None) -> dict[str, Any]:
    """Return DNS resolver configuration (`show hosts`).

    Use this to confirm the switch's name-resolution setup — whether DNS
    lookup is enabled, the default domain(s) appended to unqualified names,
    and the configured DNS server(s). Handy when NTP/AAA/syslog targets given
    as hostnames fail to resolve.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). `response.result.body` has:
      - `dnslookup` (str) — overall status line (e.g. `DNS lookup enabled`)
      - `TABLE_vrf.ROW_vrf` — a list of per-VRF resolver settings, each entry
        with `vrfname` (e.g. `default`, `management`), `defaultdomains` (the
        default domain name, e.g. `example.com`), and `domainservers` (the
        configured name servers as a whitespace/newline-separated string, e.g.
        `"  192.0.2.53\\n\\n"` — may contain leading spaces and trailing
        newlines that need trimming)
    Note the NX-API quirk: `ROW_vrf` collapses to a single object instead of a
    list when only one VRF has resolver config.
    """
    return _run_cli(switch, "show hosts")


@mcp.tool
def show_environment(switch: str | None = None) -> dict[str, Any]:
    """Return chassis environmentals (`show environment`).

    Use this for hardware health: temperature alarms, fan failures, PSU
    redundancy loss. First-line check for any "switch went down" or
    "switch overheating" alert.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). `response.result.body` groups four
    areas:
      - `powersup` — power supplies and budget:
          * `voltage_level` (int)
          * `TABLE_ps_info.ROW_ps_info` — list of PSUs: `psnum`, `model`,
            `watts`, `amps`, `status` (`Ok` / `Failed` / `Absent` / …)
          * `TABLE_mod_pow_info.ROW_mod_pow_info` — per-module draw:
            `num`, `model`, `watts_requested` / `amps_requested`,
            `watts_allocated` / `amps_allocated`, `modstatus`
          * `power_summary` — `redundancy_mode`, `redundancy_oper_mode`,
            `tot_pow_capacity`, `reserve_sup`, `available_pow`
      - `fandetails` — `TABLE_faninfo.ROW_faninfo` list: `fanname`,
        `fanmodel`, `fanhwver`, `fanstatus` (`Ok` / `Failure` / `Absent`),
        `fandirection`, `fanspeed`; plus `fan_filter_status`
      - `TABLE_temp_info.ROW_temp_info` — temperature sensors: `mod`,
        `sensor`, `cur_temp` (°C), `minor_thres` / `major_thres`,
        `alarm_status` (`Ok` / `MinorAlarm` / `MajorAlarm` / `Shutdown`)
      - `TABLE_clock_info.ROW_clock_info` — clock module: `name`, `model`,
        `status`, `act_standby`
    Note the NX-API quirk: any `ROW_*` collapses to a single object instead of
    a list when it has exactly one entry.
    """
    return _run_cli(switch, "show environment")


# ---------------------------------------------------------------------------
# Interfaces and Port-Channels (all)
# ---------------------------------------------------------------------------


@mcp.tool
def show_interface_brief(switch: str | None = None) -> dict[str, Any]:
    """Return a summary of every interface (`show interface brief`).

    Use this for quick fabric health checks: which ports are up/down, what
    VSAN they belong to, port mode (F/E/TE/NP), negotiated speed, and port-
    channel membership.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). `response.result.body` holds FOUR
    separate tables, each a `TABLE_*` / `ROW_*` list:
      - `TABLE_interface_brief_fc.ROW_interface_brief_fc` — the FC ports, each:
          * `interface` (e.g. `fc1/1`), `vsan` (int)
          * `admin_mode` (`F`/`auto`/…), `admin_trunk_mode` (`on`/`off`)
          * `status` — `up`, `notConnected`, `sfpAbsent`, `down`, …
          * `fcot_info` (SFP type, e.g. `swl`), `oper_mode`, `oper_speed`
            (Gbps), `port_channel` (bundle id or `--`), `logical_type`
            (`core`/`edge`/`--`)
          * down/empty ports show `--` for the operational fields
      - `TABLE_interface_brief_sup.ROW_interface_brief_sup` — sup ports
        (`interface`, `status`, `speed`)
      - `TABLE_interface_brief_mgmt.ROW_interface_brief_mgmt` — mgmt0
        (`interface`, `status`, `ip_address`, `speed`, `mtu`)
      - `TABLE_interface_brief_portchannel.ROW_interface_brief_portchannel` —
        port-channels (`interface`, `vsan`, `status`, `oper_mode`,
        `oper_speed`, `logical_type`)
    Note the NX-API quirk: any `ROW_*` collapses to a single object instead of
    a list when it has exactly one entry.
    """
    return _run_cli(switch, "show interface brief")


@mcp.tool
def show_interface_bbcredit(switch: str | None = None) -> dict[str, Any]:
    """Return buffer-to-buffer credit info for ALL interfaces
    (`show interface bbcredit`).

    Use this to investigate slow-drain / credit-starvation problems across the
    whole switch: when an edge device (host or array) is too slow to return BB
    credits, the switch port stalls and back-pressures the fabric. This command
    reports the CONFIGURED credit values plus each port's link state. For live
    credit-exhaustion / txwait counters use `show_interface` (per-port detail).
    For a single port use `show_interface_bbcredit_for`.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The data is deeply and IRREGULARLY
    nested: `response.result.body.TABLE_interface.ROW_interface` (a list) ->
    each row has a `TABLE_info` list of `ROW_info` groups, plus a parallel
    `flow_control` list (e.g. "R_RDY" per port). Within a `ROW_info` list the
    objects are INTERLEAVED — a credit object is followed by the
    `{interface, state}` object(s) it applies to:
      - Credit object: `transmit_b2b`, `receive_b2b`, `rx_b2b_perf_buff`,
        `rx_b2b_credit`, `tx_b2b_credit`, `tx_b2b_low_pri_cre`
      - Port object: `interface` (e.g. "fc1/1") and `state` — `"up"` for a
        live port, or a 2-element list like `["down", "SFP not present"]` /
        `["down", "Administratively down"]` / `["down", "Link failure or
        not-connected"]` for a down port (down ports carry no credit object)
    Because of this interleaving, associate each credit object with the port
    object(s) that follow it rather than assuming a 1:1 key mapping. Note the
    NX-API quirk: any `ROW_*` collapses to a single object instead of a list
    when it has exactly one entry.
    """
    return _run_cli(switch, "show interface bbcredit")


@mcp.tool
def show_interface_transceiver(switch: str | None = None) -> dict[str, Any]:
    """Return SFP / transceiver info for ALL interfaces
    (`show interface transceiver`).

    Use this to inventory every optic on the switch and spot optical-layer
    problems: missing / unsupported SFPs, marginal optical power, or
    temperature concerns. For one port use `show_interface_transceiver_for`;
    for the fuller DOM dump with alarm thresholds use
    `show_interface_transceiver_detail`.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Per-port data is at
    `response.result.body.TABLE_interface_trans.ROW_interface_trans` (a list),
    where each row has:
      - `interface_sfp` — the port (e.g. `fc1/1`)
      - `TABLE_calib.ROW_calib` — a LIST whose shape depends on presence:
          * SFP absent: a single object `{"sfp": "sfp not present"}`
          * SFP present: TWO objects — an identity object then a DOM object:
            - Identity: `sfp` ("sfp is present"), `name`, `partnum`, `rev`,
              `serialnum`, `nominal_bitrate`, `len_50` / `len_625` /
              `len_50_OM3` (m), `txcvr_type`, `tx_length`, `tx_medium`,
              `supported_speeds`, `ciscoid`, `cisco_vendor_id`,
              `cisco_part_number`, `cisco_product_id`, `cisco_version_id`,
              `xcvr_power_control`, `xcvr_power_status`
            - DOM: `temperature` (°C), `volt` (V), `current` (mA),
              `optical_tx_pwr` / `optical_rx_pwr` (dBm strings),
              `tx_fault_type`; when there is no light, intensity siblings such
              as `optical_rx_pw_intensity`, `optical_tx_pwr_intensity` or
              `curr_intensity` appear as `"--"`
    Healthy SFPs typically read RX power between -3 dBm and -10 dBm; values
    near or below -25 dBm (or `"--"`) mean no/insufficient receive light.
    Note the NX-API quirk: any `ROW_*` collapses to a single object instead of
    a list when it has exactly one entry.
    """
    return _run_cli(switch, "show interface transceiver")


@mcp.tool
def show_interface_transceiver_detail(switch: str | None = None) -> dict[str, Any]:
    """Return SFP / transceiver details with DOM and alarm/warn thresholds
    for ALL interfaces (`show interface transceiver detail`).

    The fullest optical-layer view: same identity data as
    `show_interface_transceiver`, but each present SFP also carries Digital
    Optical Monitoring (DOM) readings together with their high/low alarm and
    warning thresholds. Use this to diagnose failing SFPs, marginal optical
    power, temperature alarms, or unsupported / unlicensed optics. For a
    lighter view use `show_interface_transceiver`; for one port use
    `show_interface_transceiver_for`.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Per-port data is at
    `response.result.body.TABLE_interface_trans.ROW_interface_trans` (a list),
    where each row has:
      - `interface_sfp` — the port (e.g. `fc1/1`)
      - `TABLE_calib.ROW_calib` — a LIST whose shape depends on presence:
          * SFP absent: a single object `{"sfp": "sfp not present"}`
          * SFP present: a single object combining identity + nested DOM:
            - Identity (same fields as `show_interface_transceiver`): `sfp`
              ("sfp is present"), `name`, `partnum`, `rev`, `serialnum`,
              `nominal_bitrate`, `len_50` / `len_625` / `len_50_OM3` (m),
              `txcvr_type`, `tx_length`, `tx_medium`, `supported_speeds`,
              `ciscoid`, `cisco_vendor_id`, `cisco_part_number`,
              `cisco_product_id`, `cisco_version_id`, `xcvr_power_control`,
              `xcvr_power_status`
            - DOM with thresholds: `TABLE_calibration.ROW_calibration` ->
              `TABLE_detail.ROW_detail` (a list, normally one entry), whose
              object holds, for each metric, the live value plus its
              `*_alrm_hi` / `*_alrm_lo` (alarm) and `*_warn_hi` / `*_warn_lo`
              (warning) thresholds:
                · `temperature` (+ `temp_alrm_*` / `temp_warn_*`) — "38.05 C"
                · `voltage` (+ `volt_alrm_*` / `volt_warn_*`) — "3.29 V"
                · `current` (+ `current_alrm_*` / `current_warn_*`) — "7.50 mA"
                · `tx_pwr` (+ `tx_pwr_alrm_*` / `tx_pwr_warn_*`) — "0.29 dBm"
                · `rx_pwr` (+ `rx_pwr_alrm_*` / `rx_pwr_warn_*`) — "0.35 dBm"
                · `tx_faults` (int)
              NOTE: every value is a STRING WITH UNITS (e.g. "C", "V", "mA",
              "dBm"), not a number. When a laser has no signal, the value may
              be `"N/A"` and a sibling `*_flag` (e.g. `rx_pwr_flag`,
              `tx_pwr_flag`, `current_flag`) appears as `"--"`.
    Healthy SFPs typically read RX power between -3 dBm and -10 dBm; values
    near the `rx_pwr_alrm_lo` threshold (or ~-25 dBm / `"N/A"`) mean
    no/insufficient receive light. Note the NX-API quirk: any `ROW_*`
    collapses to a single object instead of a list when it has exactly one
    entry.
    """
    return _run_cli(switch, "show interface transceiver detail")


@mcp.tool
def show_interface_description(switch: str | None = None) -> dict[str, Any]:
    """Return the configured description of every interface
    (`show interface description`).

    Use this to read the operator-assigned labels on all ports at once (e.g.
    which host or array each port connects to). Ports with no description show
    `"--"`. For a single port use `show_interface_description_for`.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). Per-port data is at
    `response.result.body.TABLE_interface.ROW_interface` (a list), where each
    row has:
      - `interface` — the port (e.g. `fc1/1`)
      - `description` — the configured description, or `"--"` when none is set
    Note the NX-API quirk: any `ROW_*` collapses to a single object instead of
    a list when it has exactly one entry.
    """
    return _run_cli(switch, "show interface description")


@mcp.tool
def show_interface_mgmt(switch: str | None = None) -> dict[str, Any]:
    """Return detail and counters for the management interface
    (`show interface mgmt0`).

    Use this to inspect the out-of-band management port (mgmt0): its link
    state, IP address, MAC, MTU/speed, and Ethernet RX/TX packet and error
    counters. Handy when management/SSH/NX-API reachability to the switch is
    in question. This is the Ethernet mgmt port, which is why it is not
    covered by the FC per-port tools.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The data is at
    `response.result.body.TABLE_interface_mgmt.ROW_interface_mgmt` (a list,
    normally a single `mgmt0` entry), where each row has:
      - Identity / state: `interface` (e.g. `mgmt0`), `status` (`up`/`down`),
        `ip_address` (CIDR, e.g. `10.0.0.3/24`), `hardware`
        (e.g. `GigabitEthernet`), `mac_address`, `mtu` (int), `bw` (Kbps as a
        string), `bw_state` (duplex, e.g. `full`)
      - RX counters: `rx_packets`, `rx_bytes`, `rx_mcast_frames`,
        `rx_compressed`, `rx_errors`, `rx_error_frames`, `rx_dropped`,
        `rx_fifo_errors`
      - TX counters: `tx_packets`, `tx_bytes`, `tx_dropped`, `tx_errors`,
        `tx_collisions`, `tx_fifo_errors`, `tx_carrier_errors`
    Note the NX-API quirk: `ROW_interface_mgmt` collapses to a single object
    instead of a list when there is exactly one entry.
    """
    return _run_cli(switch, "show interface mgmt0")


# Port-Channels


@mcp.tool
def show_port_channel_summary(switch: str | None = None) -> dict[str, Any]:
    """Return a one-line summary of every port-channel
    (`show port-channel summary`).

    Use this for a quick overview of the ISL bundles on the switch: how many
    member ports each port-channel has and how many are currently operational.
    A port-channel whose `total_oper_ports` is less than `total_ports` has one
    or more members down/suspended — drill into the member ports with
    `show_interface_brief` or `show_interface_for` to find out why.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The port-channels are at
    `response.result.body.TABLE_port_channel_summary.ROW_port_channel_summary`
    — a list, where each entry has:
      - `interface` (str) — the bundle (note the space, e.g. `port-channel 1`)
      - `total_ports` (int) — configured member count
      - `total_oper_ports` (int) — members currently up in the bundle
      - `first_operational_port` (str) — the first member that is up (e.g.
        `fc1/1`)
    Note the NX-API quirk: `ROW_port_channel_summary` collapses to a single
    object instead of a list when there is exactly one port-channel.
    """
    return _run_cli(switch, "show port-channel summary")


@mcp.tool
def show_port_channel_usage(switch: str | None = None) -> dict[str, Any]:
    """Return which port-channel numbers are used vs free
    (`show port-channel usage`).

    Use this to pick a free port-channel id before creating a new ISL bundle,
    or to see at a glance how many of the switch's port-channel numbers are
    already allocated.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). NOTE: this command has no structured
    body — the result is a plain multi-line text block at
    `response.result.msg`, e.g. a "Totally N port-channel numbers used" header
    followed by a `Used:` line listing the allocated ids (comma-separated) and
    an `Unused:` line listing the free ids as ranges (e.g. `1 - 119`).
    """
    return _run_cli(switch, "show port-channel usage")


# ---------------------------------------------------------------------------
# Interface detail (per-port)
# ---------------------------------------------------------------------------


@mcp.tool
def show_running_config_interface_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return the running configuration of a single interface
    (`show running-config interface <interface>`).

    Use this to see exactly how ONE port is configured: admin speed, port
    mode (`F`/`E`/`auto`), VSAN membership, port-channel binding
    (`channel-group`), shutdown state, and any other interface-level config.
    Handy for auditing a port's config or comparing two ports.

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). NOTE: this command has no structured
    body — the configuration is a plain multi-line text block at
    `response.result.msg`, with leading `!Command` / `!Running configuration` /
    `!Time` comment lines and a `version` line, followed by the
    `interface <name>` stanza and its indented config lines (e.g.
    `switchport speed auto`, `switchport mode F`, `channel-group 1 force`,
    `no shutdown`).
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show running-config interface {iface}")


@mcp.tool
def show_interface_for(interface: str, switch: str | None = None) -> dict[str, Any]:
    """Return full detail for a single interface (`show interface <interface>`).

    The deep-dive view of ONE port: operational state, negotiated speed, peer
    pWWN, buffer-to-buffer credits, the full error-counter set, slow-drain
    (txwait) metrics, and the embedded transceiver/DOM readings. Use this when
    `show_interface_brief` shows a port misbehaving and you need the details.

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`; forms like `fc 1/1` are
        normalised to `fc1/1`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The interface is at
    `response.result.body.TABLE_interface.ROW_interface` — a list (normally a
    single entry), where each row includes:
      - State: `interface`, `oper_port_state` (`up`/`down`/…), `hardware`,
        `sfp`, `port_wwn`, `peer_port_wwn`, `admin_mode` / `oper_mode`,
        `admin_trunk_mode`, `port_vsan`, `admin_speed` / `oper_speed`,
        `rate_mode`, `flow_control`, `logical_type`, `bundle_if_index`
        (parent port-channel when bundled)
      - Credits: `oper_txbbcredit` / `oper_rxbbcredit`,
        `tx_b2b_credit_remain` / `rx_b2b_credit_remain`, `admin_rxbufsize`
      - Live rates: `rx_rate_bits_ps` / `tx_rate_bits_ps` (and bytes/frames)
      - Counters: `rx_frames` / `tx_frames`, `rx_bytes` / `tx_bytes`,
        `rx_invalid_crc`, `rx_error_frames`, `rx_discard_frames`,
        `rx_too_long_frames` / `rx_too_short_frames`, `tx_discard_frames`,
        `tx_error_frames`
      - Link errors: `rx_link_faliures` (NX-API's spelling), `rx_sync_loss`,
        `rx_signal_loss`, `rx_ols`/`tx_ols`, `rx_lrr`/`tx_lrr`,
        `rx_nos`/`tx_nos`, `rx_loop_inits`/`tx_loop_inits`
      - Slow drain: `txwait`, `txwait_percent_1s` / `_1m` / `_1hr` / `_72hr`,
        `tx_b2b_credit_to_zero` / `rx_b2b_credit_to_zero`, `tx_credit_loss`,
        `tx_timeout_discards`
      - `last_cleared_time` (e.g. "never")
      - `Transceiver_Info` — embedded SFP/DOM object: `serial_num`,
        `cisco_pid`, `temperature`, `voltage`, `current`, `tx_power`,
        `rx_power`, `xcvr_power_control`, `xcvr_power_status`
    Note the NX-API quirk: `ROW_interface` collapses to a single object
    instead of a list when only one interface is returned.
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show interface {iface}")


@mcp.tool
def show_interface_brief_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return the one-line summary for a single interface
    (`show interface <interface> brief`).

    Use this for a quick "is this port up?" question scoped to one
    interface. For the full chassis sweep use `show_interface_brief`.

    Args:
      interface: MDS interface name, e.g. `fc1/1` or `port-channel1`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped) with a single row:
      - `interface`, `vsan`, `admin_mode`, `oper_mode`,
        `admin_status`, `oper_status`, `oper_speed`, `port_channel`,
        `fcid`, `port_wwn` (when the port is up).
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show interface {iface} brief")


@mcp.tool
def show_interface_bbcredit_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return buffer-to-buffer credit info for a single interface
    (`show interface <interface> bbcredit`).

    Use this to investigate slow-drain / credit-starvation problems on one
    port: when an edge device (host or array) is too slow to return BB
    credits, the switch port stalls and back-pressures the fabric. This command
    reports the CONFIGURED credit values plus the port's link state. For live
    credit-exhaustion / txwait counters use `show_interface` (per-port detail).

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), scoped to the one port but with the
    same nesting as `show_interface_bbcredit`:
    `response.result.body.TABLE_interface.ROW_interface` -> `TABLE_info` ->
    `ROW_info`, where each `ROW_info` list INTERLEAVES a credit object with the
    `{interface, state}` object it applies to:
      - Credit object: `transmit_b2b`, `receive_b2b`, `rx_b2b_perf_buff`,
        `rx_b2b_credit`, `tx_b2b_credit`, `tx_b2b_low_pri_cre`
      - Port object: `interface` and `state` — `"up"` or a 2-element list like
        `["down", "SFP not present"]` (a down port carries no credit object)
    Note the NX-API quirk: any `ROW_*` collapses to a single object instead of
    a list when it has exactly one entry.
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show interface {iface} bbcredit")


@mcp.tool
def show_interface_transceiver_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return SFP / transceiver info for a single interface
    (`show interface <interface> transceiver`).

    Use this to inspect the optic plugged into ONE port (vendor, PID, serial,
    DOM readings). Lighter than `show_interface_transceiver` (all ports) or
    `show_interface_transceiver_detail` (full DOM with thresholds).

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), scoped to the one port but with the
    same nesting as `show_interface_transceiver`:
    `response.result.body.TABLE_interface_trans.ROW_interface_trans` ->
    `interface_sfp` plus `TABLE_calib.ROW_calib` (a LIST):
      - SFP absent: a single object `{"sfp": "sfp not present"}`
      - SFP present: an identity object (`name`, `partnum`, `serialnum`,
        `cisco_product_id`, `supported_speeds`, `xcvr_power_status`, …) then a
        DOM object (`temperature`, `volt`, `current`, `optical_tx_pwr`,
        `optical_rx_pwr`, `tx_fault_type`). Unlike
        `show_interface_transceiver_detail`, these DOM readings are plain
        numeric strings WITHOUT units (e.g. `"39.73"`, `"3.28"`,
        `optical_*_pwr` in dBm) and carry no alarm/warn thresholds; `"--"`
        intensity fields appear when there is no light.
    Healthy SFPs typically read RX power between -3 dBm and -10 dBm. Note the
    NX-API quirk: any `ROW_*` collapses to a single object instead of a list
    when it has exactly one entry.
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show interface {iface} transceiver")


@mcp.tool
def show_interface_transceiver_detail_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return SFP / transceiver details with DOM and alarm/warn thresholds
    for a single interface (`show interface <interface> transceiver detail`).

    The fullest optical-layer view of ONE port: identity plus Digital Optical
    Monitoring (DOM) readings with their high/low alarm and warning
    thresholds. Use this to diagnose a failing SFP, marginal optical power, or
    a temperature alarm on a specific port. For all ports use
    `show_interface_transceiver_detail`; for a lighter per-port view (no
    thresholds) use `show_interface_transceiver_for`.

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), scoped to the one port but with the
    same nesting as `show_interface_transceiver_detail`:
    `response.result.body.TABLE_interface_trans.ROW_interface_trans` ->
    `interface_sfp` plus `TABLE_calib.ROW_calib` (a LIST):
      - SFP absent: a single object `{"sfp": "sfp not present"}`
      - SFP present: a single object combining identity (`name`, `partnum`,
        `serialnum`, `cisco_product_id`, `supported_speeds`,
        `xcvr_power_status`, …) + nested DOM at
        `TABLE_calibration.ROW_calibration` -> `TABLE_detail.ROW_detail`,
        whose object holds each metric's live value plus its `*_alrm_hi` /
        `*_alrm_lo` (alarm) and `*_warn_hi` / `*_warn_lo` (warning)
        thresholds: `temperature` / `temp_*`, `voltage` / `volt_*`,
        `current` / `current_*`, `tx_pwr` / `tx_pwr_*`, `rx_pwr` / `rx_pwr_*`,
        and `tx_faults` (int). Every value is a STRING WITH UNITS (e.g.
        "39.79 C", "3.28 V", "3.60 mA", "-2.59 dBm"); an unlit laser shows
        `"N/A"` with a sibling `*_flag` of `"--"`.
    Healthy SFPs typically read RX power between -3 dBm and -10 dBm. Note the
    NX-API quirk: any `ROW_*` collapses to a single object instead of a list
    when it has exactly one entry.
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show interface {iface} transceiver detail")


@mcp.tool
def show_interface_description_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return the configured description of a single interface
    (`show interface <interface> description`).

    Use this to read the operator-assigned label on ONE port (e.g. which host
    or array it connects to). A port with no description shows `"--"`. For the
    full chassis sweep use `show_interface_description`.

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), scoped to the one port but with the
    same nesting as `show_interface_description`:
    `response.result.body.TABLE_interface.ROW_interface` -> a single row with:
      - `interface` — the port (e.g. `fc1/1`)
      - `description` — the configured description, or `"--"` when none is set
    Note the NX-API quirk: any `ROW_*` collapses to a single object instead of
    a list when it has exactly one entry.
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show interface {iface} description")


# ---------------------------------------------------------------------------
# Flogi / Name Server / Login
# ---------------------------------------------------------------------------


@mcp.tool
def show_flogi_database(switch: str | None = None) -> dict[str, Any]:
    """Return the Fabric Login (FLOGI) table (`show flogi database`).

    The FLOGI database is the authoritative list of devices currently logged
    into the local switch. Use this to answer "which port is host X plugged
    into?", "is this initiator online?", or to correlate physical cabling to
    pWWNs. A device only appears here if it is logged in to THIS switch (for
    a fabric-wide view, see `show_fcns_database`). Note that a device logged in
    over an ISL shows the `port-channel`/`fc` interface it arrived on rather
    than an edge port.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The login entries are at
    `response.result.body.TABLE_flogi_entry.ROW_flogi_entry` — a list, where
    each entry has:
      - `interface` (str) — the port the device logged in on (e.g. `fc1/1` for
        an edge port, or `port-channel1` for a device reached over an ISL)
      - `vsan` (int) — the VSAN the login is in
      - `fcid` (str) — Fibre Channel ID assigned by the fabric, hex (e.g.
        `0xab0001`)
      - `port_name` (str) — pWWN of the logged-in device
      - `node_name` (str) — nWWN of the logged-in device
      - `device_alias_for_pwwn` (str) — the device-alias mapped to `port_name`,
        present only when that pWWN resolves to one
    Alongside the table, `response.result.body.total_no_of_flogi` (int) gives
    the total login count. Note the NX-API quirk: `ROW_flogi_entry` collapses
    to a single object instead of a list when there is exactly one login.
    """
    return _run_cli(switch, "show flogi database")


@mcp.tool
def show_flogi_database_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return the FLOGI logins on a single interface
    (`show flogi database interface <interface>`).

    Scopes the Fabric Login table to ONE port, listing only the devices
    logged in through that interface. Use this to answer "what is logged in on
    fc1/1?" without dumping the whole switch with `show_flogi_database`. A
    single port commonly carries several logins (e.g. an NPIV array exposing
    multiple SVMs/pWWNs, or every host reached over an ISL `port-channel`).

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), scoped to the one port but with the
    same shape as `show_flogi_database`. The logins are at
    `response.result.body.TABLE_flogi_entry.ROW_flogi_entry` — a list, where
    each entry has:
      - `interface` (str) — the port (matches the argument)
      - `vsan` (int) — the VSAN the login is in
      - `fcid` (str) — Fibre Channel ID assigned by the fabric, hex (e.g.
        `0xab0001`)
      - `port_name` (str) — pWWN of the logged-in device
      - `node_name` (str) — nWWN of the logged-in device
      - `device_alias_for_pwwn` (str) — the device-alias mapped to `port_name`,
        present only when that pWWN resolves to one
    Alongside the table, `response.result.body.total_no_of_flogi` (int) gives
    the login count on this port. Note the NX-API quirk: `ROW_flogi_entry`
    collapses to a single object instead of a list when there is exactly one
    login.
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show flogi database interface {iface}")


@mcp.tool
def show_flogi_database_vsan(vsan: int, switch: str | None = None) -> dict[str, Any]:
    """Return the FLOGI logins in a single VSAN
    (`show flogi database vsan <vsan>`).

    Scopes the Fabric Login table to ONE VSAN, listing only the devices logged
    into the local switch in that VSAN. Use this when troubleshooting a
    specific fabric (e.g. "what is logged in to VSAN 20?") instead of dumping
    every VSAN with `show_flogi_database`.

    Args:
      vsan: VSAN id to query (integer 1-4094, e.g. `20`).
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), scoped to the one VSAN but with the
    same shape as `show_flogi_database`. The logins are at
    `response.result.body.TABLE_flogi_entry.ROW_flogi_entry` — a list, where
    each entry has:
      - `interface` (str) — the port the device logged in on (e.g. `fc1/1` for
        an edge port, or `port-channel1` for a device reached over an ISL)
      - `vsan` (int) — the VSAN the login is in (matches the argument)
      - `fcid` (str) — Fibre Channel ID assigned by the fabric, hex (e.g.
        `0xab0001`)
      - `port_name` (str) — pWWN of the logged-in device
      - `node_name` (str) — nWWN of the logged-in device
      - `device_alias_for_pwwn` (str) — the device-alias mapped to `port_name`,
        present only when that pWWN resolves to one
    Alongside the table, `response.result.body.total_no_of_flogi` (int) gives
    the login count in this VSAN. Note the NX-API quirk: `ROW_flogi_entry`
    collapses to a single object instead of a list when there is exactly one
    login.
    """
    vsan_id = _validate_vsan(vsan)
    return _run_cli(switch, f"show flogi database vsan {vsan_id}")


@mcp.tool
def show_fcns_database(switch: str | None = None) -> dict[str, Any]:
    """Return the Fibre Channel Name Server database (`show fcns database`).

    The FCNS is the FABRIC-wide directory of every device known to the SAN
    (across all switches in each VSAN). Use this to enumerate all initiators
    and targets reachable in a VSAN, regardless of which switch they are
    physically connected to.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The payload is nested per VSAN: the
    outer table is at `response.result.body.TABLE_fcns_vsan.ROW_fcns_vsan` —
    a list, where each VSAN entry has:
      - `vsan_id` (int) — the VSAN
      - `total_number_of_entries` (int) — registered devices in this VSAN
      - `TABLE_fcns_database.ROW_fcns_database` — a list of the devices, each:
          * `fcid` (str) — Fibre Channel ID, hex (e.g. `0xab0001`)
          * `type` (str) — node type (`N` = N-port, `NL` = NL-port)
          * `pwwn` (str) — port WWN
          * `vendor` (str) — e.g. `NetApp`, `Cisco`
          * `device_alias` (str) — the device-alias for `pwwn`, present only
            when it resolves to one
          * `TABLE_fc4_type_feature.ROW_fc4_type_feature` — present only when
            the device registered FC4 features; a list of `{fc4_type,
            fc4_feature}` pairs, e.g. `scsi-fcp` / `target`, `NVMe` /
            `target,disc`, `fc-av` / `target`, or `npv` / `""`. Use these to
            distinguish initiators from targets and SCSI-FCP from NVMe-oF.
    Compare to `show_flogi_database` to determine whether a device is local to
    this switch or learned via an ISL. Note the NX-API quirk: any `ROW_*`
    (`ROW_fcns_vsan`, `ROW_fcns_database`, `ROW_fc4_type_feature`) collapses to
    a single object instead of a list when it has exactly one entry.
    """
    return _run_cli(switch, "show fcns database")


@mcp.tool
def show_fcns_database_vsan(vsan: int, switch: str | None = None) -> dict[str, Any]:
    """Return the Name Server database for one VSAN
    (`show fcns database vsan <vsan>`).

    Like `show_fcns_database` but scoped to a single VSAN, so the response
    isn't cluttered with the other fabrics. Use this to enumerate the
    initiators and targets registered in one specific VSAN.

    Args:
      vsan: VSAN id to query (integer 1-4094, e.g. `20`).
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), with the same nesting as
    `show_fcns_database` but normally a single VSAN entry. The outer table is
    at `response.result.body.TABLE_fcns_vsan.ROW_fcns_vsan` — a list, where the
    VSAN entry has:
      - `vsan_id` (int) — matches the argument
      - `total_number_of_entries` (int) — registered devices in this VSAN
      - `TABLE_fcns_database.ROW_fcns_database` — a list of the devices, each:
          * `fcid` (str) — Fibre Channel ID, hex (e.g. `0xab0001`)
          * `type` (str) — node type (`N` = N-port, `NL` = NL-port)
          * `pwwn` (str) — port WWN
          * `vendor` (str) — e.g. `NetApp`, `Cisco`
          * `device_alias` (str) — the device-alias for `pwwn`, present only
            when it resolves to one
          * `TABLE_fc4_type_feature.ROW_fc4_type_feature` — present only when
            the device registered FC4 features; a list of `{fc4_type,
            fc4_feature}` pairs, e.g. `scsi-fcp` / `target`, `NVMe` /
            `target,disc`, `fc-av` / `target`, or `npv` / `""`. Use these to
            distinguish initiators from targets and SCSI-FCP from NVMe-oF.
    Note the NX-API quirk: any `ROW_*` (`ROW_fcns_vsan`, `ROW_fcns_database`,
    `ROW_fc4_type_feature`) collapses to a single object instead of a list when
    it has exactly one entry.
    """
    vsan_id = _validate_vsan(vsan)
    return _run_cli(switch, f"show fcns database vsan {vsan_id}")


@mcp.tool
def show_fcns_database_detail(switch: str | None = None) -> dict[str, Any]:
    """Return the FULL Name Server entry for every device
    (`show fcns database detail`).

    The verbose form of `show_fcns_database`: in addition to the summary
    fields it returns each device's nWWN, FC4 types/features, symbolic
    port/node names (which reveal the array/SVM/UCS identity), the fabric port
    WWN, and the local switch/interface the device is attached to. Use this
    when you need the rich per-device detail — e.g. mapping a pWWN to a NetApp
    SVM LIF or a UCS san-port-channel — rather than just a flat listing.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The outer table is at
    `response.result.body.TABLE_fcns_vsan.ROW_fcns_vsan` — a list. NOTE: in the
    detail form each device gets its OWN `ROW_fcns_vsan` entry (so `vsan_id`
    repeats), and `total_number_of_entries` (int) appears only on the LAST
    entry. Each `ROW_fcns_vsan` has:
      - `vsan_id` (int)
      - `TABLE_fcns_database.ROW_fcns_database` — normally a single device
        object, with:
          * Identity: `fcid` (hex, e.g. `0xab0001`), `pwwn`, `nwwn`,
            `vendor`, `device_alias` (when it resolves), `port_type`
            (`N`/`NL`), `class`
          * FC4: `TABLE_fc4_types_fc4_features.ROW_fc4_types_fc4_features` —
            present only when registered; a list of `{fc4_types, fc4_features}`
            pairs (note these key names differ from the non-detail command's
            `fc4_type`/`fc4_feature`), e.g. `scsi-fcp`/`target`,
            `NVMe`/`target,disc`, `npv`/`""`
          * Symbolic names: `symbolic_port_name`, `symbolic_node_name` (e.g.
            the NetApp adapter/SVM LIF or UCS `san-port-channel` string)
          * Addressing: `node_ip_addr`, `port_ip_addr`, `ipa`, `hard_addr`,
            `fabric_port_wwn`, `permanent_pwwn`, `permanent_vendor`
          * Attachment: `connected_interface` (e.g. `fc1/1` or
            `port-channel1`), `switch_name`, `switch_ip`
    Note the NX-API quirk: any `ROW_*` collapses to a single object instead of
    a list when it has exactly one entry.
    """
    return _run_cli(switch, "show fcns database detail")


# ---------------------------------------------------------------------------
# VSAN
# ---------------------------------------------------------------------------


@mcp.tool
def show_vsan(switch: str | None = None) -> dict[str, Any]:
    """Return the list of VSANs and their state (`show vsan`).

    Use this to discover which VSANs exist on the switch, their operational
    state, and load-balancing policy. Many other commands are VSAN-scoped,
    so this is a good first-call when the operator hasn't specified a VSAN.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The VSANs are at
    `response.result.body.TABLE_vsan.ROW_vsan` — a list, where each entry has
    `vsan` (int, the VSAN id). For a normal VSAN the entry also has:
      - `name` (str) — VSAN name (e.g. `VSAN0001`, `vsan-prod-a`)
      - `state` (str) — admin state (`active` / `suspended`)
      - `operational_state` (str) — `up` / `down` (a VSAN with no member
        ports/ISLs reads `down`)
      - `interop_mode` (str) — e.g. `default`, `interop-1`
      - `load_balancing` (str) — e.g. `src-id/dst-id/oxid`
    The reserved/special VSANs instead carry a single name field and none of
    the above: the EVFP control VSAN (e.g. 4079) has `evfp_control_vsan_name`
    (e.g. `evfp_isolated_vsan`), and the isolated VSAN (e.g. 4094) has
    `inactive_vsan_name` (e.g. `isolated_vsan`) — the isolated VSAN is where
    the switch quarantines ports. Note the NX-API quirk: `ROW_vsan` collapses
    to a single object instead of a list when only one VSAN is returned.
    """
    return _run_cli(switch, "show vsan")


@mcp.tool
def show_vsan_id(vsan: int, switch: str | None = None) -> dict[str, Any]:
    """Return the state of a single VSAN (`show vsan <vsan>`).

    Like `show_vsan` but scoped to one VSAN id, so the response isn't
    cluttered with the other VSANs. Use this to check a specific fabric's
    name, admin/operational state and load-balancing policy.

    Args:
      vsan: VSAN id to query (integer 1-4094, e.g. `20`).
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), with the same shape as `show_vsan` but
    normally a single VSAN entry. The VSAN is at
    `response.result.body.TABLE_vsan.ROW_vsan` — a list, where the entry has
    `vsan` (int, matches the argument). For a normal VSAN the entry also has:
      - `name` (str) — VSAN name (e.g. `vsan-prod-a`)
      - `state` (str) — admin state (`active` / `suspended`)
      - `operational_state` (str) — `up` / `down` (a VSAN with no member
        ports/ISLs reads `down`)
      - `interop_mode` (str) — e.g. `default`, `interop-1`
      - `load_balancing` (str) — e.g. `src-id/dst-id/oxid`
    The reserved/special VSANs instead carry a single name field and none of
    the above: the EVFP control VSAN (e.g. 4079) has `evfp_control_vsan_name`,
    and the isolated VSAN (e.g. 4094) has `inactive_vsan_name`. Note the
    NX-API quirk: `ROW_vsan` collapses to a single object instead of a list
    when only one VSAN is returned.
    """
    vsan_id = _validate_vsan(vsan)
    return _run_cli(switch, f"show vsan {vsan_id}")


@mcp.tool
def show_vsan_usage(switch: str | None = None) -> dict[str, Any]:
    """Return how many VSANs are configured and which ids are free
    (`show vsan usage`).

    Use this to see at a glance how many VSANs exist and, crucially, which
    VSAN ids are still available before creating a new one — the counterpart
    to `show_port_channel_usage` but for VSANs.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). `response.result.body` holds three
    scalar fields:
      - `no_of_vsan_configured` (int) — count of configured VSANs
      - `configured_range_of_vsans` (str) — the configured ids as a
        comma-separated list/ranges (e.g. `1,20`)
      - `vsans_available_to_configure` (str) — the free ids as
        comma-separated ranges (e.g. `2-19,21-4078,4080-4093`); note the
        reserved ids (e.g. 4079 EVFP, 4094 isolated) are excluded
    """
    return _run_cli(switch, "show vsan usage")


@mcp.tool
def show_vsan_membership(switch: str | None = None) -> dict[str, Any]:
    """Return which interfaces belong to each VSAN (`show vsan membership`).

    Use this to see the port-to-VSAN mapping for the whole switch: which FC
    ports and port-channels are assigned to each VSAN. Complements `show_vsan`
    (which lists the VSANs themselves) when you need to know where a VSAN's
    member ports are.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The data is at
    `response.result.body.TABLE_vsan_membership.ROW_vsan_membership` — a list,
    one entry per VSAN, where each entry has:
      - `vsan` (int) — the VSAN id
      - `interfaces` — the member ports (e.g. `fc1/1`, `port-channel1`). This
        is a LIST of port strings, but collapses to a single STRING when the
        VSAN has exactly one member (e.g. `"fc1/6"`); absent when the VSAN has
        no members.
    The reserved/special VSANs also carry a name field: the EVFP control VSAN
    (e.g. 4079) has `vsan_evfp_control_name` (e.g. `evfp_isolated_vsan`) and
    the isolated VSAN (e.g. 4094) has `vsan_inactive_name` (e.g.
    `isolated_vsan`) — note these key names differ from `show_vsan`'s
    `evfp_control_vsan_name` / `inactive_vsan_name`. Note the NX-API quirk:
    `ROW_vsan_membership` collapses to a single object instead of a list when
    only one VSAN is returned.
    """
    return _run_cli(switch, "show vsan membership")


@mcp.tool
def show_vsan_membership_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return the VSAN membership of a single interface
    (`show vsan membership interface <interface>`).

    Scopes the port-to-VSAN mapping to ONE interface: which VSAN the port
    belongs to and, for a trunk, the list of VSANs allowed on it. Use this to
    answer "what VSAN is fc1/3 in?" without dumping every port with
    `show_vsan_membership`.

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), scoped to the one port. The data is at
    `response.result.body.TABLE_vsan_membership_interface.ROW_vsan_membership_interface`
    — a list (normally a single entry), where each entry has:
      - `name` (str) — the interface (e.g. `fc1/3`)
      - `vsan` (int) — the port VSAN it belongs to
      - `allowed_vsan_list` (str) — for a trunk, the VSANs allowed on the port
        as comma-separated ranges (e.g. `1-4078,4080-4093`)
    Note the NX-API quirk: `ROW_vsan_membership_interface` collapses to a
    single object instead of a list when only one interface is returned.
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show vsan membership interface {iface}")


# ---------------------------------------------------------------------------
# Interface Counters
# ---------------------------------------------------------------------------


@mcp.tool
def show_interface_counters(switch: str | None = None) -> dict[str, Any]:
    """Return per-interface FC error counters (`show interface counters`).

    THE first command to run on any link-quality complaint (flapping ports,
    SCSI timeouts, slow drain). Surfaces low-level errors that
    `show_interface_brief` hides. For a single port use
    `show_interface_counters_for`.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The per-interface counters are at
    `response.result.body.TABLE_counters.ROW_counters` — a list. The rows have
    TWO shapes depending on the port kind. FC ports and port-channels carry the
    FC counter set, each with:
      - `interface` (e.g. `fc1/1`, `port-channel1`)
      - Live rates: `rx_rate_bits_ps` / `rx_rate_bytes_ps` /
        `rx_rate_frames_ps` and the `tx_rate_*` equivalents
      - Totals: `rx_frames` / `tx_frames`, `rx_bytes` / `tx_bytes`, plus a
        per-class breakdown `rx_c2_frames`/`rx_c2_bytes`,
        `rx_c3_frames`/`rx_c3_bytes`, `rx_cf_frames`/`rx_cf_bytes` (and the
        `tx_c2_*` / `tx_c3_*` / `tx_cf_*` equivalents)
      - Errors: `rx_crc_fcs` (CRC/FCS errors), `rx_discard_frames`,
        `rx_error_frames`, `rx_unknown_class_frames`, `rx_too_long_frames`,
        `rx_too_short_frames`, `tx_discard_frames`, `tx_error_frames`
      - Forwarding drops: `pg_acl_drops`, `pg_fib_drops`, `pg_xbar_drops`,
        `pg_other_drops` (each with a `*_start` / `*_end` packet-group range)
      - Link errors: `rx_link_faliures` (NX-API's spelling), `rx_sync_loss`,
        `rx_signal_loss`, `rx_ols`/`tx_ols`, `rx_lrr`/`tx_lrr`,
        `rx_nos`/`tx_nos`, `rx_loop_inits`/`tx_loop_inits`
      - Slow drain / credits: `txwait`, `txwait_percent_1s` / `_1m` / `_1hr` /
        `_72hr`, `tx_b2b_credit_to_zero` / `rx_b2b_credit_to_zero`,
        `tx_timeout_discards`, `tx_credit_loss`, `rx_b2b_credit_remain` /
        `tx_b2b_credit_remain`, `tx_b2b_low_pri_cre`
      - `last_cleared_time` (e.g. "never") — present on FC ports; port-channels
        omit `rx_b2b_credit_remain` / `tx_b2b_credit_remain` /
        `tx_b2b_low_pri_cre` / `last_cleared_time`
    The supervisor / management ports (`sup-fc0`, `mgmt0`) instead carry an
    Ethernet-style set: `rx_packets`, `rx_bytes`, `rx_mcast_frames`,
    `rx_compressed`, `rx_errors`, `rx_error_frames`, `rx_overrun`, `rx_fifo`,
    `tx_packets`, `tx_bytes`, `tx_underruns`, `tx_errors`, `tx_collisions`,
    `tx_fifo`, `tx_carrier_errors`.
    Non-zero `rx_crc_fcs` / `rx_sync_loss` / `rx_link_faliures` → suspect a
    dirty fibre, bad SFP, or wrong cable type; pair with
    `show_interface_transceiver_detail`. Note the NX-API quirk: `ROW_counters`
    collapses to a single object instead of a list when only one interface is
    returned.
    """
    return _run_cli(switch, "show interface counters")


@mcp.tool
def show_interface_counters_brief(switch: str | None = None) -> dict[str, Any]:
    """Return a compact per-interface frame/rate summary
    (`show interface counters brief`).

    The lightweight counterpart to `show_interface_counters`: instead of the
    full FC error-counter set it returns just the live rate and total frame
    counts per port. Use this for a quick "which ports are passing traffic?"
    sweep; drill into a busy or suspect port with `show_interface_counters_for`
    for the detailed errors.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). `response.result.body` holds TWO
    separate tables, each a `TABLE_*` / `ROW_*` list whose rows share the same
    fields:
      - `TABLE_counters_brief.ROW_counters_brief` — the FC ports
      - `TABLE_port_channel.ROW_port_channel` — the port-channels
    Each row has:
      - `interface` (str) — the port (e.g. `fc1/1`, `port-channel1`)
      - `rx_rate_MBps` / `tx_rate_MBps` (int) — current RX/TX rate in MB/s
      - `rx_total_frames` / `tx_total_frames` (int) — cumulative RX/TX frames
    Note the NX-API quirk: each `ROW_*` collapses to a single object instead of
    a list when it has exactly one entry.
    """
    return _run_cli(switch, "show interface counters brief")


@mcp.tool
def show_interface_counters_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return error / frame counters for a single interface
    (`show interface <interface> counters`).

    Use this when the operator asks about ONE specific port (e.g. "any errors
    on fc1/1?"). For a fabric-wide sweep use `show_interface_counters` instead.

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), scoped to the one port but with the
    same shape as `show_interface_counters`. The counters are at
    `response.result.body.TABLE_counters.ROW_counters` — a list (normally a
    single entry for an FC port / port-channel), where each row has:
      - `interface` (matches the argument)
      - Live rates: `rx_rate_bits_ps` / `rx_rate_bytes_ps` /
        `rx_rate_frames_ps` and the `tx_rate_*` equivalents
      - Totals: `rx_frames` / `tx_frames`, `rx_bytes` / `tx_bytes`, plus the
        per-class breakdown `rx_c2_*` / `rx_c3_*` / `rx_cf_*` (and `tx_c2_*` /
        `tx_c3_*` / `tx_cf_*`)
      - Errors: `rx_crc_fcs`, `rx_discard_frames`, `rx_error_frames`,
        `rx_unknown_class_frames`, `rx_too_long_frames`, `rx_too_short_frames`,
        `tx_discard_frames`, `tx_error_frames`
      - Forwarding drops: `pg_acl_drops`, `pg_fib_drops`, `pg_xbar_drops`,
        `pg_other_drops` (each with a `*_start` / `*_end` range)
      - Link errors: `rx_link_faliures` (NX-API's spelling), `rx_sync_loss`,
        `rx_signal_loss`, `rx_ols`/`tx_ols`, `rx_lrr`/`tx_lrr`,
        `rx_nos`/`tx_nos`, `rx_loop_inits`/`tx_loop_inits`
      - Slow drain / credits: `txwait`, `txwait_percent_1s` / `_1m` / `_1hr` /
        `_72hr`, `tx_b2b_credit_to_zero` / `rx_b2b_credit_to_zero`,
        `tx_timeout_discards`, `tx_credit_loss`, `rx_b2b_credit_remain` /
        `tx_b2b_credit_remain`, `tx_b2b_low_pri_cre`
      - `last_cleared_time` (e.g. "never")
    Note the NX-API quirk: `ROW_counters` collapses to a single object instead
    of a list when only one interface is returned (the usual case here).
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show interface {iface} counters")


@mcp.tool
def show_interface_counters_brief_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return the compact frame/rate summary for a single interface
    (`show interface <interface> counters brief`).

    The lightweight per-port counterpart to `show_interface_counters_for`:
    just the live rate and total frame counts for ONE port, with none of the
    detailed error counters. For the full chassis sweep use
    `show_interface_counters_brief`.

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), scoped to the one port. The data is at
    `response.result.body.TABLE_counters_brief.ROW_counters_brief` — a list
    (normally a single entry), where each row has:
      - `interface` (str) — the port (matches the argument)
      - `rx_rate_MBps` / `tx_rate_MBps` (int) — current RX/TX rate in MB/s
      - `rx_total_frames` / `tx_total_frames` (int) — cumulative RX/TX frames
    Note the NX-API quirk: `ROW_counters_brief` collapses to a single object
    instead of a list when only one interface is returned (the usual case
    here).
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show interface {iface} counters brief")


@mcp.tool
def show_interface_counters_detailed_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return the fully grouped counter set for a single interface
    (`show interface <interface> counters detailed`).

    The richest per-port counter view: the same metrics as
    `show_interface_counters_for` but organised into named sub-tables (rate,
    totals, link, congestion, others) and with extra fields not in the flat
    form. Use this when you need every counter for ONE port — frame totals by
    class, link integrity errors, FEC, slow-drain/credit and forwarding-drop
    detail. For the flat single-table form use `show_interface_counters_for`;
    for a quick rate/frame summary use `show_interface_counters_brief_for`.

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`), scoped to the one port. The data is at
    `response.result.body.TABLE_counters.ROW_counters` — a list (normally a
    single entry), where each row has `interface` (matches the argument) plus
    FIVE nested `TABLE_*` / `ROW_*` groups (each itself a single-entry list):
      - `TABLE_rate.ROW_rate` — live rates: `rx_rate_bits_ps` /
        `tx_rate_bits_ps`, `rx_rate_bytes_ps` / `tx_rate_bytes_ps`,
        `rx_rate_frames_ps` / `tx_rate_frames_ps`
      - `TABLE_totals.ROW_totals` — frame/byte totals: `rx_frames` /
        `tx_frames`, `rx_bytes` / `tx_bytes`, `rx_mcast_frames` /
        `tx_mcast_frames`, `rx_bcast_frames` / `tx_bcast_frames`,
        `rx_ucast_frames` / `tx_ucast_frames`, `rx_discard_frames` /
        `tx_discard_frames`, `rx_error_frames` / `tx_error_frames`, plus the
        per-class breakdown `rx_c2_*` / `tx_c2_*` (incl.
        `rx_c2_discard_frames`, `rx_c2_port_rjt_frames`), `rx_c3_*` / `tx_c3_*`
        (incl. `rx_c3_discard_frames`) and `rx_cf_*` / `tx_cf_*` (incl.
        `rx_cf_discard_frames`)
      - `TABLE_link.ROW_link` — link integrity: `rx_link_failures` (NOTE:
        spelled correctly here, unlike the flat command's `rx_link_faliures`),
        `rx_sync_loss`, `rx_signal_loss`, `rx_prm_seq_pro_err`,
        `rx_inv_trans_err`, `rx_inv_crc`, `rx_delim_err`, `rx_frag_frames`,
        `rx_eof_abort_frames`, `rx_unknown_class_frames`, `rx_runt_frames`,
        `rx_jabber_frames`, `rx_too_long_frames`, `rx_too_short_frames`,
        `rx_fec_corrected`, `rx_fec_uncorrected`, `rx_link_reset` /
        `tx_link_reset`, `rx_link_reset_resp` / `tx_link_reset_resp`,
        `rx_off_seq_err` / `tx_off_seq_err`, `rx_non_oper_seq` /
        `tx_non_oper_seq`, `bb_scs_resend`, `bb_scr_incr`
      - `TABLE_congestion.ROW_congestion` — slow-drain/credits:
        `tx_timeout_discards`, `tx_credit_loss`, `txwait`,
        `txwait_percent_1s` / `_1m` / `_1hr` / `_72hr`, `rx_b2b_credit_remain`
        / `tx_b2b_credit_remain`, `tx_b2b_low_pri_cre`, `rx_b2b_credit_to_zero`
        / `tx_b2b_credit_to_zero`
      - `TABLE_others.ROW_others` — forwarding drops + clear time:
        `pg_acl_drops`, `pg_fib_drops` (+ `pg_fib_start` / `pg_fib_end`),
        `pg_xbar_drops` (+ `pg_xbar_start` / `pg_xbar_end`), `pg_other_drops`,
        `last_cleared_time` (e.g. "never")
    Note the NX-API quirk: any `ROW_*` collapses to a single object instead of
    a list when it has exactly one entry (the usual case for every group here).
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show interface {iface} counters detailed")


@mcp.tool
def show_interface_aggregate_counters_for(
    interface: str, switch: str | None = None
) -> dict[str, Any]:
    """Return cumulative-since-clear counters for a single interface
    (`show interface <interface> aggregate-counters`).

    Use this for long-term trending (errors accumulated since the counters
    were last cleared with `clear counters`). Pair with
    `show_interface_counters_for` for the live snapshot — a divergence
    suggests recent vs historical degradation.

    Args:
      interface: MDS interface name, e.g. `fc1/1`, `fc1/1/1` or
        `port-channel1`. Validated against
        `^(fc\\d+/\\d+(?:/\\d+)?|port-channelN)$`.
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The data is at
    `response.result.body.TABLE_interface.ROW_interface` — a list (normally a
    single entry). NOTE: unlike `show_interface_counters_for`, every field name
    here carries an `_aggr` suffix and the interface is under `str_aggr` (NOT
    `interface`). Each row has:
      - `str_aggr` (str) — the interface (e.g. `fc1/3`)
      - Rates: `in_bps_aggr` / `in_byps_aggr` / `in_fps_aggr` and the
        `out_bps_aggr` / `out_byps_aggr` / `out_fps_aggr` equivalents
      - Totals: `total_in_frames_aggr` / `total_out_frames_aggr`,
        `total_in_bytes_aggr` / `total_out_bytes_aggr`, plus a per-class
        breakdown `C2InFrames_aggr`/`C2InOctets_aggr`,
        `C3InFrames_aggr`/`C3InOctets_aggr`, `CfInFrames_aggr`/`CfInOctets_aggr`
        (and the `*Out*` equivalents)
      - Errors: `total_in_discards_aggr` / `total_out_discards_aggr`,
        `total_in_errors_aggr` / `total_out_errors_aggr`, `InvalidCrcs_aggr`,
        `UnknownClassFrames_aggr`, `FramesTooLong_aggr`, `FramesTooShort_aggr`
      - Link errors: `LinkFailures_aggr`, `SyncLosses_aggr`, `SigLosses_aggr`,
        `OlsIns_aggr`/`OlsOuts_aggr`, `LRRIn_aggr`/`LRROut_aggr`,
        `NOSIn_aggr`/`NOSOut_aggr`, `in_lip_aggr`/`out_lip_aggr`
      - Credits: `TxBBCreditTransistionToZero_aggr` /
        `RxBBCreditTransistionToZero_aggr` (NX-API's spelling of
        "Transition"), `rx_b2b_credit`, `tx_b2b_credit`, `tx_b2b_low_pri_cre`
    Note the NX-API quirk: `ROW_interface` collapses to a single object instead
    of a list when only one interface is returned (the usual case here).
    """
    iface = _validate_interface(interface)
    return _run_cli(switch, f"show interface {iface} aggregate-counters")


# ---------------------------------------------------------------------------
# Sessions / Users
# ---------------------------------------------------------------------------


@mcp.tool
def show_users(switch: str | None = None) -> dict[str, Any]:
    """Return the currently logged-in users (`show users`).

    Use this for an active-session audit — see who is connected, from which
    source IP, over which line, how long they have been idle, and the login
    process PID (useful to `clear line` a stuck session). Good first check when
    investigating a config change or an unexpected concurrent edit.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The active sessions are at
    `response.result.body.TABLE_sessions.ROW_sessions` — a list, where each
    entry has:
      - `name` (str) — the logged-in username (e.g. `admin`)
      - `line` (str) — the tty/pts line (e.g. `pts/7`)
      - `time` (str) — login time (e.g. `Jan 01 00:00`)
      - `idle` (str) — idle duration (e.g. `00:04`, or `.` when active now)
      - `pid` (int) — the session process id
      - `comment` (str) — source and protocol (e.g.
        `(192.0.2.10) session=ssh`)
    The same `name` can appear on multiple lines when a user has several
    concurrent sessions. Note the NX-API quirk: `ROW_sessions` collapses to a
    single object instead of a list when only one user is logged in.
    """
    return _run_cli(switch, "show users")


@mcp.tool
def show_accounting_log(switch: str | None = None) -> dict[str, Any]:
    """Return the command accounting log (`show accounting log`).

    Use this for a security/change audit trail — every login/logout and every
    configuration command run on the switch, with who ran it, from where, and
    whether it succeeded. Complements `show_users` (current sessions only) by
    giving the full historical record. Also captures failed authentications.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). The log lives at
    `response.result.body.TABLE_acctlog_time.ROW_acctlog_time` — a list whose
    entries each hold a single `accountlog_starttime` (str) field. IMPORTANT:
    this is NOT one row per event — NX-API splits the whole log into ~8 KB text
    CHUNKS, and a chunk boundary can fall in the MIDDLE of a line. To get the
    full log, concatenate every `accountlog_starttime` value in order, THEN
    split the result on newlines (`\\n`). Each resulting line is one event,
    typically formatted:
      `<timestamp>:type=<start|stop|update>:id=<src>@<line>:user=<user>:cmd=<command>`
    where:
      - `<timestamp>` — e.g. `Mon Jan 01 00:00:00 2025`
      - `type` — `start` (login/session begin), `stop` (logout/termination),
        or `update` (a command was executed)
      - `id` — `<source>@<line>`, e.g. `192.0.2.10@pts/2` (SSH from an IP on a
        pty), `192.0.2.10@ssh.22071`, or `192.0.2.10@nginx.5443` (NX-API/web)
      - `user` — the account name (e.g. `admin`)
      - `cmd` — the command or event text; config commands end with
        `(SUCCESS)` / `(FAILURE)`, and session ends read e.g.
        `shell terminated because the ssh session closed` or
        `shell termination forced`
    Failed logins appear as a special line:
      `<timestamp>:type=update:id=::PAM:user= Authentication failed from <ip>`.
    Note the NX-API quirk: `ROW_acctlog_time` collapses to a single object
    instead of a list when the log is small enough to fit in one chunk.
    """
    return _run_cli(switch, "show accounting log")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@mcp.tool
def show_logging(switch: str | None = None) -> dict[str, Any]:
    """Return the most recent syslog messages (`show logging last 200`).

    Use this for forensic / time-correlation work: almost every fault on the
    switch leaves a trail in syslog. Always cross-reference timestamps with
    the operator's reported incident time.

    Args:
      switch: Name of the MDS switch to query (see `list_switches`). May be
        omitted only when a single switch is configured.

    Returns the raw NX-API JSON-RPC response (wrapped under `response`, with
    the queried switch under `switch`). NOTE: this command has no structured
    body — the log is a plain multi-line text block at `response.result.msg`
    (NX-API does not structure syslog), one message per line, each typically
    formatted:
      `<timestamp> <hostname> %<FACILITY>-<severity>-<MNEMONIC>: <message>`
    The timestamp has no year-trailing comma and the severity is the DIGIT embedded
    in the `%FACILITY-N-MNEMONIC` tag (not a separate field); some lines insert an
    extra process token between the hostname and the `%` tag (e.g.
    `... mds-fab-a _f worker %AUTHPRIV-3-SYSTEM_MSG: ...`).

    Severity levels: 0=emergency, 1=alert, 2=critical, 3=error, 4=warning,
    5=notification, 6=informational, 7=debugging. Filter for level <= 4 to
    spot real problems.

    Common SAN-relevant mnemonics: `PORT-5-IF_DOWN_LINK_FAILURE`,
    `ZONE-2-ZS_MERGE_FAILED`, `FSPF-3-NBR_DOWN`, `MODULE-2-MOD_DIAG_FAIL`,
    `PFM-2-FAN_FAILED`, `PLATFORM-2-PS_FAIL`.
    """
    return _run_cli(switch, "show logging last 200")


# ---------------------------------------------------------------------------
# Entrypoint — selects the MCP transport based on MCP_TRANSPORT.
#   * "stdio" : used when the server is launched as a subprocess by an MCP
#               client. Communicates over stdin/stdout.
#   * "http"  : used when the server runs as a standalone, network-reachable
#               process. Exposes the MCP endpoint on
#               http://<host>:<port>/mcp/.
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the FastMCP server, choosing the transport from MCP_TRANSPORT.

    Runs the module-level `mcp` (a `FastMCP` instance) over either stdio or
    streamable HTTP. Defaults to HTTP on MCP_HOST:MCP_PORT.
    """
    transport = os.getenv("MCP_TRANSPORT", "http").lower()
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport="http",
            host=os.getenv("MCP_HOST", "0.0.0.0"),
            port=int(os.getenv("MCP_PORT", "8000")),
        )


if __name__ == "__main__":
    main()
