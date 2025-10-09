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

function ts() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

const PREFIX = `[codex-postinstall ${ts()} pid:${process.pid}]`;
function info(msg) { console.log(`${PREFIX} ${msg}`); }
function warn(msg) { console.warn(`${PREFIX} WARN: ${msg}`); }
function error(msg) { console.error(`${PREFIX} ERROR: ${msg}`); }

function statSafe(p) {
  try { return fs.statSync(p); } catch (_) { return null; }
}

function modeStr(mode) {
  if (typeof mode !== 'number') return 'n/a';
  return '0' + (mode & 0o7777).toString(8);
}

function runCapture(cmd, args, opts = {}) {
  const start = Date.now();
  const res = spawnSync(cmd, args, { shell: false, encoding: 'utf-8', ...opts });
  const dur = Date.now() - start;
  const out = {
    ok: res.status === 0,
    status: res.status,
    signal: res.signal || null,
    stdout: res.stdout || '',
    stderr: res.stderr || '',
    duration_ms: dur,
    error: res.error || null,
  };
  info(`run: ${cmd} ${args.join(' ')} [status=${out.status} dur=${out.duration_ms}ms]`);
  if (out.stdout) process.stdout.write(`${PREFIX} stdout: ${out.stdout}`);
  if (out.stderr) process.stderr.write(`${PREFIX} stderr: ${out.stderr}`);
  if (out.error) error(`spawn error: ${out.error && out.error.message ? out.error.message : String(out.error)}`);
  return out;
}

function run(cmd, args, opts = {}) {
  const res = runCapture(cmd, args, { stdio: 'pipe', ...opts });
  return res.ok;
}

function existsOnPath(bin) {
  if (process.platform === 'win32') {
    const res = runCapture('where', [`${bin}.cmd`]);
    return res.ok;
  }
  const res = runCapture('command', ['-v', bin]);
  return res.ok;
}

function codexAvailable() {
  const onPath = existsOnPath('codex');
  info(`check codex on PATH: ${onPath}`);
  if (!onPath) return false;
  const ver = runCapture('codex', ['--version']);
  return ver.ok;
}

function ensureAuthFromEnvIfMissing() {
  const home = os.homedir() || process.env.HOME || process.env.USERPROFILE;
  if (!home) {
    warn('Could not resolve home directory; skip ~/.codex/auth.json bootstrap');
    return;
  }

  const codexDir = path.join(home, '.codex');
  const authPath = path.join(codexDir, 'auth.json');

  const authExists = fs.existsSync(authPath);
  info(`auth path: ${authPath} exists=${authExists}`);
  if (authExists) return; // already present; nothing to do

  const secret = process.env.CODEX_AUTH_JSON;
  if (!secret || String(secret).trim().length === 0) {
    info('CODEX_AUTH_JSON not set; skipping auth.json creation');
    return;
  }

  try {
    info(`creating ${codexDir}`);
    fs.mkdirSync(codexDir, { recursive: true });
    // Write exactly the provided content without trailing newline, restrict permissions
    const bytes = Buffer.byteLength(String(secret), 'utf-8');
    fs.writeFileSync(authPath, String(secret), { mode: 0o600 });
    try { fs.chmodSync(authPath, 0o600); } catch (_) { /* best-effort on non-POSIX */ }
    const st = statSafe(authPath);
    info(`wrote auth to ${authPath} size=${bytes}B mode=${st ? modeStr(st.mode) : 'n/a'}`);
  } catch (err) {
    error(`Failed to write ~/.codex/auth.json: ${err && err.message ? err.message : String(err)}\n${err && err.stack ? err.stack : ''}`);
    process.exit(1);
  }
}

function ensureConfigTomlFromRepo() {
  // Mirrors the behavior in .github/workflows/tuning.yml: fail fast if repo config is missing.
  const repoConfigPath = path.join(process.cwd(), '.codex', 'config.toml');
  info(`repo config candidate: ${repoConfigPath}`);
  if (!fs.existsSync(repoConfigPath)) {
    error('::error::.codex/config.toml not found in repo; aborting per fail-fast policy');
    process.exit(2);
  }

  const home = os.homedir() || process.env.HOME || process.env.USERPROFILE;
  if (!home) {
    error('HOME not set; cannot locate ~/.codex for config.toml');
    process.exit(1);
  }

  const codexDir = path.join(home, '.codex');
  const destConfigPath = path.join(codexDir, 'config.toml');
  try {
    info(`ensure ~/.codex dir: ${codexDir}`);
    fs.mkdirSync(codexDir, { recursive: true });
    // Force overwrite to keep local config in sync with repo baseline
    const content = fs.readFileSync(repoConfigPath);
    info(`read repo config bytes=${content.length}`);
    fs.writeFileSync(destConfigPath, content, { mode: 0o600 });
    try { fs.chmodSync(destConfigPath, 0o600); } catch (_) { /* best-effort on non-POSIX */ }
    const st = statSafe(destConfigPath);
    info(`installed config to ${destConfigPath} size=${content.length}B mode=${st ? modeStr(st.mode) : 'n/a'}`);
  } catch (err) {
    error(`Failed to install ~/.codex/config.toml: ${err && err.message ? err.message : String(err)}\n${err && err.stack ? err.stack : ''}`);
    process.exit(1);
  }
}

function installGlobalCodexWithEnv(extraEnv = {}) {
  const env = { ...process.env, ...extraEnv };
  info(`attempting npm global install @openai/codex@latest with env delta keys=[${Object.keys(extraEnv).join(', ')}]`);
  const res = runCapture('npm', ['install', '-g', '@openai/codex@latest'], { env });
  return res.ok;
}

function main() {
  info('=== BEGIN codex postinstall ===');
  info(`node: ${process.version} platform: ${process.platform} arch: ${process.arch}`);
  info(`cwd: ${process.cwd()}`);
  info(`shell: ${process.env.SHELL || 'n/a'}`);
  info(`PATH: ${process.env.PATH}`);
  const whichNode = runCapture(process.platform === 'win32' ? 'where' : 'which', ['node']);
  const whichNpm = runCapture(process.platform === 'win32' ? 'where' : 'which', ['npm']);
  const npmPrefix = runCapture('npm', ['config', 'get', 'prefix']);
  const npmBinG = runCapture('npm', ['bin', '-g']);

  // Always attempt to populate auth.json if missing and env var is provided
  // (this runs regardless of Codex presence).
  ensureAuthFromEnvIfMissing();
  // Always install/refresh config.toml from the repo; fail-fast if missing
  ensureConfigTomlFromRepo();

  if (codexAvailable()) {
    info('codex already available; skipping installation');
    info('=== END codex postinstall (noop) ===');
    return;
  }

  // Codex is missing and needs installation; proceed with install attempts.

  // First attempt: default global prefix
  info('codex not found; starting installation attempts');
  if (!installGlobalCodexWithEnv()) {
    // Retry with user-level prefix without mutating global npm config
    const home = os.homedir() || process.env.HOME || process.env.USERPROFILE;
    if (!home) {
      error('HOME not set; cannot retry with user prefix');
    } else {
      const userPrefix = `${home}/.npm-global`;
      const env = { NPM_CONFIG_PREFIX: userPrefix };
      warn(`Retrying global install with user prefix at ${userPrefix}`);
      info(`will add PATH hint if installation succeeds (prefix/bin)`);
      if (!installGlobalCodexWithEnv(env)) {
        error('Global installation failed with user prefix');
      }
      // Try to run codex from that prefix explicitly to validate
      if (!codexAvailable()) {
        // Try executing the binary directly from the expected bin folder to confirm installation
        const binPath = `${userPrefix}/bin/codex`;
        const ok = run(binPath, ['--version']);
        if (!ok) {
          error('Codex CLI still not available on PATH after installation attempts.');
          error('Ensure your npm global bin is on PATH, or install manually:');
          error('  npm install -g @openai/codex@latest');
          process.exit(1);
          return;
        } else {
          // Not on PATH, but installed; print guidance
          warn(`Codex installed at ${binPath} but not found on PATH.`);
          warn('Add the following to your shell profile to use `codex` globally:');
          warn(`  export PATH=\"${userPrefix}/bin:$PATH\"`);
        }
        return;
      }
    }
  }

  if (!codexAvailable()) {
    error('Codex CLI not found after installation.');
    process.exit(1);
  }
  const ver = runCapture('codex', ['--version']);
  if (ver.ok) info('codex installation verified');
  info('=== END codex postinstall ===');
}

main();
