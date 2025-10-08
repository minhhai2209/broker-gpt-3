#!/usr/bin/env node
/*
  Ensures a global Codex CLI is available as `codex`.
  - Tries `codex --version`; if missing, installs globally via `npm install -g @openai/codex@latest`.
  - If installation is needed, ensure ~/.codex/auth.json exists; if missing, populate with $CODEX_AUTH_JSON (same contract as .github/workflows/tuning.yml).
  - On EACCES or missing PATH to global bin, retries with a user prefix using NPM_CONFIG_PREFIX=$HOME/.npm-global.
  - Verifies availability and fails fast if still not found.

  Fail-fast policy: if ~/.codex/auth.json is required but missing and $CODEX_AUTH_JSON is not provided, exit with a clear error.
*/

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, { stdio: 'inherit', shell: false, ...opts });
  return res.status === 0;
}

function existsOnPath(bin) {
  const res = spawnSync(process.platform === 'win32' ? 'where' : 'command', [process.platform === 'win32' ? bin + '.cmd' : '-v', bin], { stdio: 'ignore', shell: false });
  return res.status === 0;
}

function codexAvailable() {
  return existsOnPath('codex') && run('codex', ['--version']);
}

function ensureAuthFromEnvIfMissing() {
  const home = os.homedir() || process.env.HOME || process.env.USERPROFILE;
  if (!home) {
    console.warn('[codex-postinstall] Could not resolve home directory; skip ~/.codex/auth.json bootstrap');
    return;
  }

  const codexDir = path.join(home, '.codex');
  const authPath = path.join(codexDir, 'auth.json');

  if (fs.existsSync(authPath)) {
    return; // already present; nothing to do
  }

  const secret = process.env.CODEX_AUTH_JSON;
  if (!secret || String(secret).trim().length === 0) {
    // Env not provided; silently skip per relaxed policy.
    return;
  }

  try {
    fs.mkdirSync(codexDir, { recursive: true });
    // Write exactly the provided content without trailing newline, restrict permissions
    fs.writeFileSync(authPath, String(secret), { mode: 0o600 });
    try { fs.chmodSync(authPath, 0o600); } catch (_) { /* best-effort on non-POSIX */ }
    console.log(`[codex-postinstall] Wrote auth to ${authPath}`);
  } catch (err) {
    console.error('[codex-postinstall] Failed to write ~/.codex/auth.json:', err && err.message ? err.message : String(err));
    process.exit(1);
  }
}

function ensureConfigTomlFromRepo() {
  // Mirrors the behavior in .github/workflows/tuning.yml: fail fast if repo config is missing.
  const repoConfigPath = path.join(process.cwd(), '.codex', 'config.toml');
  if (!fs.existsSync(repoConfigPath)) {
    console.error('::error::.codex/config.toml not found in repo; aborting per fail-fast policy');
    process.exit(2);
  }

  const home = os.homedir() || process.env.HOME || process.env.USERPROFILE;
  if (!home) {
    console.error('[codex-postinstall] HOME not set; cannot locate ~/.codex for config.toml');
    process.exit(1);
  }

  const codexDir = path.join(home, '.codex');
  const destConfigPath = path.join(codexDir, 'config.toml');
  try {
    fs.mkdirSync(codexDir, { recursive: true });
    // Force overwrite to keep local config in sync with repo baseline
    const content = fs.readFileSync(repoConfigPath);
    fs.writeFileSync(destConfigPath, content, { mode: 0o600 });
    try { fs.chmodSync(destConfigPath, 0o600); } catch (_) { /* best-effort on non-POSIX */ }
    console.log(`[codex-postinstall] Installed config to ${destConfigPath}`);
  } catch (err) {
    console.error('[codex-postinstall] Failed to install ~/.codex/config.toml:', err && err.message ? err.message : String(err));
    process.exit(1);
  }
}

function installGlobalCodexWithEnv(extraEnv = {}) {
  const env = { ...process.env, ...extraEnv };
  return run('npm', ['install', '-g', '@openai/codex@latest'], { env });
}

function main() {
  // Always attempt to populate auth.json if missing and env var is provided
  // (this runs regardless of Codex presence).
  ensureAuthFromEnvIfMissing();
  // Always install/refresh config.toml from the repo; fail-fast if missing
  ensureConfigTomlFromRepo();

  if (codexAvailable()) {
    return;
  }

  // Codex is missing and needs installation; proceed with install attempts.

  // First attempt: default global prefix
  if (!installGlobalCodexWithEnv()) {
    // Retry with user-level prefix without mutating global npm config
    const home = os.homedir() || process.env.HOME || process.env.USERPROFILE;
    if (!home) {
      console.error('[codex-postinstall] HOME not set; cannot retry with user prefix');
    } else {
      const userPrefix = `${home}/.npm-global`;
      const env = { NPM_CONFIG_PREFIX: userPrefix };
      console.warn(`[codex-postinstall] Retrying global install with user prefix at ${userPrefix}`);
      if (!installGlobalCodexWithEnv(env)) {
        console.error('[codex-postinstall] Global installation failed with user prefix');
      }
      // Try to run codex from that prefix explicitly to validate
      if (!codexAvailable()) {
        // Try executing the binary directly from the expected bin folder to confirm installation
        const binPath = `${userPrefix}/bin/codex`;
        const ok = run(binPath, ['--version']);
        if (!ok) {
          console.error('[codex-postinstall] Codex CLI still not available on PATH after installation attempts.');
          console.error('[codex-postinstall] Ensure your npm global bin is on PATH, or install manually:');
          console.error('  npm install -g @openai/codex@latest');
          process.exit(1);
          return;
        } else {
          // Not on PATH, but installed; print guidance
          console.warn(`[codex-postinstall] Codex installed at ${binPath} but not found on PATH.`);
          console.warn('[codex-postinstall] Add the following to your shell profile to use `codex` globally:');
          console.warn(`  export PATH=\"${userPrefix}/bin:$PATH\"`);
        }
        return;
      }
    }
  }

  if (!codexAvailable()) {
    console.error('[codex-postinstall] Codex CLI not found after installation.');
    process.exit(1);
  }
}

main();
