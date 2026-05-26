"""
Synthetic security-advisory corpus for chunk 13 (e09 cross-domain
validation).

12 short advisory-style summaries on classic appsec vulnerability
classes. Deliberately synthetic (not real CVEs) so the experiment is
fully reproducible — anyone replicating runs identical content with
no link rot, no vendor takedowns, no judge-model contamination from
solver pretraining (the synthetic version uses fictional vendor
names, fake CVE IDs, and stylized claims).

The corpus is designed with deliberate coverage + deliberate gaps,
exactly the same way `e03_production_traffic/corpus.py` is structured:

  Coverage (12 docs, two per fictional vendor so cross-doc consistency
  tasks have natural pairings):
    - Acme Systems / AcmeCMS:      buffer overflow + stored XSS
    - Quartz Software / QuartzDB:  SQL injection + log-format RCE
    - Helios Networks / Gateway:   SSRF + DoS-via-recursion
    - Skyline Corp / Identity:     JWT bypass + weak PRNG
    - Polaris Tech / Vault:        local priv-esc + TOCTOU race
    - Beacon Labs / Mesh:          supply-chain dependency + IDOR

  Deliberate gaps (so Class-5 / no-answer-in-corpus tasks have real
  ground truth):
    - Kernel / OS-internal vulnerabilities (none)
    - Hardware side-channels (Spectre/Meltdown family) (none)
    - Cloud IAM misconfigurations (none)
    - Firmware / pre-boot supply-chain (none)
    - Network-layer attacks (BGP hijacking, DNS rebinding) (none)

  Text length: ~200-280 words each, comfortable for the memory agent's
  context window.

The advisory format follows the public-CVE convention loosely:
  ID, Title, Affected (product + version range), Reporter, Disclosed
  date, CVSS score + vector, Description (technical), Impact (what
  the attacker gains), Mitigation (patch + workaround).
"""

from __future__ import annotations

from agensflow import Document


def get_corpus_subset(doc_ids: list[str]) -> list[Document]:
    """Resolve a list of doc IDs against CORPUS. Mirrors e03's helper of
    the same name so SecurityTask.documents can use the same shape."""
    by_id = {d.id: d for d in CORPUS}
    missing = [i for i in doc_ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown corpus doc id(s): {missing}")
    return [by_id[i] for i in doc_ids]


CORPUS: list[Document] = [
    # ============================================================== #
    # DOC 1 — Acme Systems / AcmeCMS — stack buffer overflow
    # ============================================================== #
    Document(
        id="cve-demo-001-buffer-overflow",
        text=(
            "CVE-DEMO-2026-001 — Stack buffer overflow in AcmeCMS content "
            "import parser.\n\n"
            "Affected: AcmeCMS 7.2.0 through 7.4.6 (inclusive). The "
            "long-term-support branch 6.x is not affected because the "
            "vulnerable parser was introduced in the 7.2.0 rewrite.\n\n"
            "Reporter: J. Mendoza, independent researcher.\n"
            "Disclosed: 2025-04-18.\n"
            "CVSS: 7.5 (HIGH) — vector AV:N/AC:H/PR:L/UI:N/S:U/C:H/I:H/A:H.\n\n"
            "Description. The `parse_import_block()` function in AcmeCMS's "
            "content-import subsystem copies attacker-supplied block names "
            "into a 256-byte stack buffer using strcpy(). The size of the "
            "attacker-supplied block name is bounded only by the import "
            "file's outer structure; crafted imports can supply names up "
            "to 8KB. Because the import endpoint accepts uploads from any "
            "authenticated editor, exploitation requires only a low-"
            "privileged account and a single HTTP POST with a malicious "
            "import payload.\n\n"
            "Impact. Reliable remote code execution as the AcmeCMS service "
            "user. The vulnerable binary on stock Linux distributions "
            "ships without stack canaries due to a build-flag oversight, "
            "and ASLR alone is insufficient to prevent exploitation; a "
            "working public exploit has been demonstrated.\n\n"
            "Mitigation. Upgrade to AcmeCMS 7.4.7 or later, which "
            "replaces the strcpy() call with a bounded copy. As a "
            "workaround, administrators can disable the content-import "
            "endpoint by setting `cms.import.enabled = false` in "
            "`acmecms.conf` and restarting the service. Multi-factor "
            "authentication does not prevent exploitation since the "
            "attacker need only authenticate as any editor."
        ),
    ),

    # ============================================================== #
    # DOC 2 — Quartz Software / QuartzDB — SQL injection
    # ============================================================== #
    Document(
        id="cve-demo-002-sql-injection",
        text=(
            "CVE-DEMO-2026-002 — Authenticated SQL injection in QuartzDB "
            "REST query endpoint.\n\n"
            "Affected: QuartzDB Server 4.0.0 through 4.3.2 when the "
            "optional `/v1/query/raw` REST endpoint is enabled. The "
            "endpoint is enabled by default in evaluation builds and "
            "disabled by default in production builds, but the "
            "documentation in versions 4.0–4.2 incorrectly stated the "
            "inverse.\n\n"
            "Reporter: S. Tanaka, Quartz Security Team (internal).\n"
            "Disclosed: 2025-07-09.\n"
            "CVSS: 7.6 (HIGH) — vector AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:N.\n\n"
            "Description. The `/v1/query/raw` endpoint accepts a JSON "
            "body containing a filter expression and constructs the "
            "downstream SQL by concatenating the filter string directly "
            "into a `WHERE` clause. The endpoint was intended only for "
            "internal use by the QuartzDB CLI, which sanitizes filters "
            "client-side; the server performs no sanitization or "
            "parameterization. Any authenticated user with the "
            "`query.execute` permission can supply arbitrary SQL via the "
            "filter expression.\n\n"
            "Impact. Full read and write access to all tables visible to "
            "the query-executing service account, including the audit "
            "log. Attackers who escalate to the audit log can cover their "
            "tracks. Because QuartzDB uses a single shared service "
            "account for all REST traffic, even users without direct "
            "database privileges can reach administrative tables.\n\n"
            "Mitigation. Upgrade to QuartzDB Server 4.3.3 or later, "
            "which rewrites the endpoint to use parameterized queries. "
            "Operators who cannot upgrade immediately should disable "
            "`/v1/query/raw` via the configuration flag "
            "`rest.raw_query.enabled = false`. Audit logs should be "
            "reviewed for unexpected query patterns originating from "
            "the REST service account."
        ),
    ),

    # ============================================================== #
    # DOC 3 — Acme Systems / AcmeCMS — stored XSS in comments
    # ============================================================== #
    Document(
        id="cve-demo-003-xss-stored",
        text=(
            "CVE-DEMO-2026-003 — Stored cross-site scripting in AcmeCMS "
            "comment renderer.\n\n"
            "Affected: AcmeCMS 7.0.0 through 7.4.8. The 6.x LTS branch "
            "is also affected from 6.5.0 onward, when the "
            "comment-attachment feature was backported.\n\n"
            "Reporter: J. Mendoza, independent researcher.\n"
            "Disclosed: 2025-08-30.\n"
            "CVSS: 6.4 (MEDIUM) — vector AV:N/AC:L/PR:L/UI:R/S:C/C:L/I:L/A:N.\n\n"
            "Description. The AcmeCMS comment renderer sanitizes the "
            "comment body but fails to sanitize the `data-attachment-url` "
            "attribute on inline attachment references. An attacker with "
            "a registered commenter account can submit a comment that "
            "renders an attachment whose URL contains a `javascript:` "
            "scheme; when an admin or editor later previews the comment "
            "in the moderation queue, the attribute is rendered into the "
            "page DOM verbatim and the script executes in the admin's "
            "browser session.\n\n"
            "Impact. Session hijacking against any AcmeCMS admin or "
            "editor who previews a malicious comment. Because the "
            "moderation queue auto-loads comments on page open, simply "
            "navigating to the queue is sufficient to trigger the "
            "payload — no explicit user action is required beyond "
            "visiting the moderation UI. From a hijacked admin session, "
            "an attacker can install plugins, modify content, or "
            "exfiltrate user data.\n\n"
            "Mitigation. Upgrade to AcmeCMS 7.4.9 or LTS 6.7.4, which "
            "sanitize attachment URLs through a strict allowlist of "
            "`http`, `https`, and `mailto` schemes. As a workaround, "
            "administrators can disable the inline-attachment feature "
            "by setting `comments.allow_attachments = false`. Multi-"
            "factor authentication on admin accounts does not prevent "
            "exploitation since the attacker rides the admin's already-"
            "established session."
        ),
    ),

    # ============================================================== #
    # DOC 4 — Helios Networks / Helios Gateway — SSRF to cloud metadata
    # ============================================================== #
    Document(
        id="cve-demo-004-ssrf-cloud",
        text=(
            "CVE-DEMO-2026-004 — Server-side request forgery in Helios "
            "Gateway URL-rewrite rules permits cloud-metadata-service "
            "access.\n\n"
            "Affected: Helios Gateway 3.0.0 through 3.4.2. The 2.x "
            "branch is end-of-life and does not include the URL-rewrite "
            "subsystem.\n\n"
            "Reporter: A. Volkov, Quartet Security Research.\n"
            "Disclosed: 2025-03-11.\n"
            "CVSS: 8.5 (HIGH) — vector AV:N/AC:L/PR:L/UI:N/S:C/C:H/I:L/A:N.\n\n"
            "Description. The URL-rewrite engine in Helios Gateway "
            "follows redirects when resolving target URLs but does not "
            "validate the resolved destination against a configured "
            "deny-list. An authenticated tenant operator can craft a "
            "rewrite rule whose target initially points to a benign "
            "external host that redirects to the cloud provider's "
            "instance-metadata service at `169.254.169.254`. Subsequent "
            "Gateway requests against the rewritten target retrieve "
            "instance credentials, including the IAM role's temporary "
            "session tokens.\n\n"
            "Impact. Cross-tenant credential exposure. A single "
            "tenant's operator can obtain the cloud-instance IAM role "
            "credentials shared across all tenants on the same Gateway "
            "fleet, escalating from one-tenant operator to fleet-wide "
            "cloud access. The vector's Scope:Changed reflects this "
            "crossing of authorization boundaries.\n\n"
            "Mitigation. Upgrade to Helios Gateway 3.4.3 or later, "
            "which adds a default deny-list covering RFC1918, "
            "link-local, and known metadata addresses. Operators can "
            "also set `gateway.rewrite.deny_link_local = true` in the "
            "configuration. Per-tenant IAM roles, rather than a shared "
            "fleet role, eliminate the cross-tenant impact even if the "
            "SSRF itself remains exploitable."
        ),
    ),

    # ============================================================== #
    # DOC 5 — Skyline Corp / Skyline Identity — JWT signature bypass
    # ============================================================== #
    Document(
        id="cve-demo-005-auth-bypass-jwt",
        text=(
            "CVE-DEMO-2026-005 — JWT signature verification bypass in "
            "Skyline Identity allows forged authentication tokens.\n\n"
            "Affected: Skyline Identity 5.1.0 through 5.6.3. Earlier "
            "5.0.x releases use a different JWT library and are not "
            "affected.\n\n"
            "Reporter: L. Okafor, NorthStar Labs.\n"
            "Disclosed: 2025-09-22.\n"
            "CVSS: 9.1 (CRITICAL) — vector AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N.\n\n"
            "Description. Skyline Identity's token validator accepts JWTs "
            "whose `alg` header is set to `none`, despite documentation "
            "stating that only `RS256` and `ES256` are accepted. The "
            "vulnerability stems from a configuration parser that "
            "treats the algorithm allowlist as a comma-separated string "
            "but silently splits on whitespace as well, so the value "
            "`'RS256, ES256'` is interpreted as `['RS256,', 'ES256']` "
            "and the literal `'none'` value bypasses the allowlist "
            "comparison entirely because of an unrelated null-check fallback.\n\n"
            "Impact. Any unauthenticated attacker can forge a JWT "
            "asserting any user identity, including administrator "
            "accounts. No signing key is required. Skyline Identity acts "
            "as the SSO front-end for most Skyline products, so a forged "
            "token grants downstream access to every Skyline service "
            "trusting that identity provider.\n\n"
            "Mitigation. Upgrade to Skyline Identity 5.6.4 or later, "
            "which rejects `alg=none` regardless of allowlist parsing "
            "and adds an integration test covering the exact bypass. "
            "Operators who cannot upgrade immediately can deploy the "
            "vendor-provided WAF rule that drops requests carrying "
            "tokens with `alg=none`."
        ),
    ),

    # ============================================================== #
    # DOC 6 — Polaris Tech / Polaris Vault — local privilege escalation
    # ============================================================== #
    Document(
        id="cve-demo-006-priv-esc-local",
        text=(
            "CVE-DEMO-2026-006 — Local privilege escalation in Polaris "
            "Vault setuid helper.\n\n"
            "Affected: Polaris Vault 2.8.0 through 2.10.4 on Linux. The "
            "Windows and macOS builds use a different privilege model "
            "and are not affected.\n\n"
            "Reporter: M. Patel, Polaris Tech Internal Red Team.\n"
            "Disclosed: 2025-02-04.\n"
            "CVSS: 7.8 (HIGH) — vector AV:L/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H.\n\n"
            "Description. Polaris Vault ships a setuid-root helper "
            "binary `vault-mount` that mounts encrypted volumes on "
            "behalf of unprivileged users. The helper does not drop "
            "supplementary group privileges before invoking the mount "
            "syscall, and it uses an environment variable to locate the "
            "FUSE driver. A local attacker can supply a `LD_PRELOAD`-"
            "like environment override that points to an attacker-"
            "controlled shared library, which is loaded with root "
            "permissions because the helper retains the effective root "
            "UID through the load.\n\n"
            "Impact. Any local user on a system with Polaris Vault "
            "installed can obtain root privileges. Exploitation does "
            "not require any Polaris Vault account or volume; the "
            "setuid bit on the helper is the only precondition.\n\n"
            "Mitigation. Upgrade to Polaris Vault 2.10.5 or later, "
            "which sets the secure-execution flag on the helper binary "
            "and unsets sensitive environment variables before "
            "executing privileged operations. A workaround is to "
            "remove the setuid bit from `/usr/lib/polaris-vault/vault-"
            "mount` (volume mounting will require manual root "
            "intervention until the patched release is installed)."
        ),
    ),

    # ============================================================== #
    # DOC 7 — Beacon Labs / Beacon Mesh — malicious transitive dependency
    # ============================================================== #
    Document(
        id="cve-demo-007-supply-chain-dep",
        text=(
            "CVE-DEMO-2026-007 — Supply-chain compromise via malicious "
            "transitive dependency in Beacon Mesh build pipeline.\n\n"
            "Affected: Beacon Mesh 1.2.0 through 1.4.1. Builds produced "
            "between 2025-05-14 and 2025-06-02 contain the malicious "
            "code; builds from before or after that window are clean "
            "but should still be upgraded.\n\n"
            "Reporter: R. Lindqvist, Beacon Labs Security.\n"
            "Disclosed: 2025-06-05.\n"
            "CVSS: 8.3 (HIGH) — vector AV:N/AC:H/PR:N/UI:R/S:C/C:H/I:H/A:H.\n\n"
            "Description. Between 2025-05-14 and 2025-06-02, the "
            "package registry mirror used by Beacon Mesh's CI pipeline "
            "was serving a typosquatted version of an indirect "
            "dependency (`tiny-yaml-parser`) whose name differs from "
            "the legitimate `tiny_yaml_parser` only in the underscore. "
            "The malicious version exfiltrates build-time environment "
            "variables — including registry tokens and signing keys — "
            "to an attacker-controlled host before performing the "
            "advertised parsing operation, so the build itself succeeds "
            "and downstream consumers receive a backdoored Mesh "
            "binary.\n\n"
            "Impact. Any deployment of Beacon Mesh whose binary was "
            "built during the affected window may have leaked CI "
            "secrets and may itself contain backdoor code, including a "
            "reverse-shell trigger on a specific HTTP header value. "
            "Affected operators should rotate all secrets that were "
            "present in the CI environment during the window.\n\n"
            "Mitigation. Upgrade to Beacon Mesh 1.4.2 or later, which "
            "pins the legitimate `tiny_yaml_parser` (with the "
            "underscore) and adds a registry-source allowlist to the "
            "CI configuration. Operators must additionally rotate any "
            "secrets that were present during the affected window; "
            "binary replacement alone is insufficient."
        ),
    ),

    # ============================================================== #
    # DOC 8 — Skyline Corp / Skyline Identity — weak PRNG (pair-link with 005)
    # ============================================================== #
    Document(
        id="cve-demo-008-crypto-weak-rng",
        text=(
            "CVE-DEMO-2026-008 — Insufficiently random session-token "
            "generation in Skyline Identity.\n\n"
            "Affected: Skyline Identity 5.0.0 through 5.6.3 — the same "
            "branch family as CVE-DEMO-2026-005 but with a wider "
            "vulnerable range (the weak PRNG predates the JWT subsystem).\n\n"
            "Reporter: L. Okafor, NorthStar Labs.\n"
            "Disclosed: 2025-10-14.\n"
            "CVSS: 5.9 (MEDIUM) — vector AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:N/A:N.\n\n"
            "Description. Skyline Identity seeds its session-token "
            "PRNG using the process start time in seconds, masked with "
            "a 16-bit installation identifier. The total entropy of an "
            "active-session token is therefore approximately 30 bits, "
            "well below the 128 bits required for unguessable session "
            "identifiers. An attacker who can observe several valid "
            "session tokens (e.g., from a logged-in admin) can "
            "reconstruct the seed and predict tokens for subsequent "
            "sessions issued by the same Identity process.\n\n"
            "Impact. Session prediction allows an attacker to hijack "
            "active sessions without credential theft. When combined "
            "with CVE-DEMO-2026-005, the impact is amplified: the "
            "attacker can both forge tokens (via the `alg=none` bypass) "
            "and predict legitimate tokens, depending on which path "
            "the downstream service trusts. Operators have reported "
            "successful end-to-end attack chains using both CVEs in "
            "sequence.\n\n"
            "Mitigation. Upgrade to Skyline Identity 5.6.4 or later, "
            "which replaces the custom PRNG with the platform's "
            "cryptographically-secure random source. There is no "
            "supported workaround short of upgrading; reducing token "
            "lifetime only narrows the attack window."
        ),
    ),

    # ============================================================== #
    # DOC 9 — Helios Networks / Helios Gateway — DoS via unbounded recursion
    # ============================================================== #
    Document(
        id="cve-demo-009-dos-resource-exhaustion",
        text=(
            "CVE-DEMO-2026-009 — Denial of service via unbounded "
            "recursion in Helios Gateway routing-table reload.\n\n"
            "Affected: Helios Gateway 3.0.0 through 3.4.6 — the same "
            "3.x branch as CVE-DEMO-2026-004 but extending two point "
            "releases further because the routing-table fix landed "
            "later than the SSRF fix.\n\n"
            "Reporter: A. Volkov, Quartet Security Research.\n"
            "Disclosed: 2025-08-18.\n"
            "CVSS: 7.5 (HIGH) — vector AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H.\n\n"
            "Description. The Gateway routing-table parser resolves "
            "route-group references recursively without a depth limit. "
            "A crafted configuration in which group `A` references "
            "group `B`, group `B` references group `C`, and group `C` "
            "references group `A`, causes the parser to recurse "
            "indefinitely on next reload, exhausting the process stack "
            "and crashing the Gateway. Reload is triggered by any "
            "valid administrator action; the SIGHUP signal is also "
            "sufficient.\n\n"
            "Impact. A single malformed configuration push crashes the "
            "Gateway across the fleet on next reload. The crash is "
            "deterministic and persists across restarts until the "
            "configuration is rolled back, producing a sustained "
            "outage rather than a transient one. No authentication "
            "is required if administrators apply configuration files "
            "from an untrusted source.\n\n"
            "Mitigation. Upgrade to Helios Gateway 3.4.7 or later, "
            "which caps recursion depth at 64 and reports a "
            "configuration error rather than crashing. Operators can "
            "also enforce configuration linting before each push; the "
            "vendor publishes a `helios-config-lint` binary that "
            "detects the recursion pattern."
        ),
    ),

    # ============================================================== #
    # DOC 10 — Polaris Tech / Polaris Vault — TOCTOU race
    # ============================================================== #
    Document(
        id="cve-demo-010-race-toctou",
        text=(
            "CVE-DEMO-2026-010 — TOCTOU race in Polaris Vault file-"
            "permission check during volume export.\n\n"
            "Affected: Polaris Vault 2.7.0 through 2.10.4 — overlapping "
            "but slightly wider than CVE-DEMO-2026-006 (the race "
            "predates the setuid helper rework).\n\n"
            "Reporter: M. Patel, Polaris Tech Internal Red Team.\n"
            "Disclosed: 2025-05-23.\n"
            "CVSS: 5.9 (MEDIUM) — vector AV:L/AC:H/PR:L/UI:N/S:U/C:H/I:H/A:N.\n\n"
            "Description. The volume-export routine checks that the "
            "caller has read permission on a source file by calling "
            "`access()` and then opens the file with `open()`. A local "
            "attacker can replace the source path with a symlink "
            "between the check and the open, redirecting the export "
            "to a file the attacker cannot legitimately read (e.g., "
            "another user's vault file). Both this and CVE-DEMO-2026-006 "
            "stem from the same code-review oversight: privileged "
            "filesystem operations are guarded by permission checks "
            "that do not account for race conditions or environment "
            "manipulation.\n\n"
            "Impact. Disclosure of files readable only by other "
            "Polaris Vault users on the same host, including "
            "encrypted-volume key files. Exploitation requires "
            "winning the race; reliability is approximately 30% per "
            "attempt on stock kernels.\n\n"
            "Mitigation. Upgrade to Polaris Vault 2.10.5 or later, "
            "which replaces the `access()`-then-`open()` pattern with "
            "`openat()` plus an `fstat()` permission verification on "
            "the resulting descriptor. No workaround short of "
            "upgrading; restricting volume export to administrators "
            "narrows the attack surface but does not close it."
        ),
    ),

    # ============================================================== #
    # DOC 11 — Beacon Labs / Beacon Mesh — IDOR exposing other tenants
    # ============================================================== #
    Document(
        id="cve-demo-011-idor-api",
        text=(
            "CVE-DEMO-2026-011 — Insecure direct object reference in "
            "Beacon Mesh telemetry API permits cross-tenant resource "
            "enumeration.\n\n"
            "Affected: Beacon Mesh 1.3.0 through 1.4.3 — overlapping "
            "with the supply-chain advisory CVE-DEMO-2026-007 in the "
            "1.3.x and 1.4.x lines.\n\n"
            "Reporter: R. Lindqvist, Beacon Labs Security.\n"
            "Disclosed: 2025-11-02.\n"
            "CVSS: 6.5 (MEDIUM) — vector AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N.\n\n"
            "Description. The Mesh telemetry API exposes per-resource "
            "metrics under URIs of the form `/v2/telemetry/<resource-"
            "uuid>`. The server resolves the UUID against the global "
            "resource table and returns the metrics, but the tenancy "
            "check that should restrict results to the calling user's "
            "own tenant is performed only in the UI gateway, not in "
            "the API layer. Any authenticated tenant user who guesses "
            "or enumerates UUIDs from other tenants can read those "
            "tenants' resource telemetry directly via the API.\n\n"
            "Impact. Cross-tenant disclosure of resource-level metrics "
            "(request rates, response sizes, error patterns), which "
            "can in turn reveal customer identities, traffic patterns, "
            "and business-sensitive operational data. Write operations "
            "are unaffected because they enforce tenancy at the API "
            "layer separately.\n\n"
            "Mitigation. Upgrade to Beacon Mesh 1.4.4 or later, which "
            "adds the tenancy check to the API layer to match the UI "
            "gateway. As a workaround, operators can deploy the "
            "vendor-provided sidecar policy that injects a tenancy "
            "filter into telemetry-API responses; the sidecar is "
            "available for all 1.3.x and 1.4.x releases."
        ),
    ),

    # ============================================================== #
    # DOC 12 — Quartz Software / QuartzDB — log-format RCE
    # ============================================================== #
    Document(
        id="cve-demo-012-log-injection-rce",
        text=(
            "CVE-DEMO-2026-012 — Remote code execution via log-format "
            "expansion in QuartzDB structured logger.\n\n"
            "Affected: QuartzDB Server 4.1.0 through 4.3.4. The 3.x "
            "branch uses a different logging library and is not "
            "affected.\n\n"
            "Reporter: S. Tanaka, Quartz Security Team (internal).\n"
            "Disclosed: 2025-12-01.\n"
            "CVSS: 8.8 (HIGH) — vector AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:H/A:H.\n\n"
            "Description. QuartzDB's structured logger interprets a "
            "`${expr}` syntax inside any log-line input as a dynamic "
            "lookup, which can resolve to outbound JNDI/LDAP-style "
            "URLs and load remote class definitions. The vulnerable "
            "format expansion was added to support templated log "
            "messages but was never disabled for user-supplied input. "
            "Any authenticated user who can submit a query whose error "
            "message includes their input — for example, a malformed "
            "filter expression — can trigger the expansion when the "
            "server logs the error.\n\n"
            "Impact. Reliable remote code execution as the QuartzDB "
            "service user. As in CVE-DEMO-2026-002, the executed code "
            "runs under the shared service account used for all REST "
            "traffic, so the impact is consistent across that "
            "advisory's architectural assumption: any RCE in QuartzDB "
            "reaches every table visible to the service account.\n\n"
            "Mitigation. Upgrade to QuartzDB Server 4.3.5 or later, "
            "which disables format expansion in the logger by default. "
            "Operators can also set the system property "
            "`quartzdb.logger.format_lookups = false` and restart, or "
            "remove the affected library from the classpath if running "
            "an older release that cannot be upgraded immediately."
        ),
    ),
]

