#!/usr/bin/env node
/*
  Ensures a global Codex CLI is available as `codex`.
  - Tries `codex --version`; if missing, installs globally via `npm install -g @openai/codex@latest`.
  - On EACCES or missing PATH to global bin, retries with a user prefix using NPM_CONFIG_PREFIX=$HOME/.npm-global.
  - Verifies availability and fails fast if still not found.
*/

const { spawnSync } = require('child_process');

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

function installGlobalCodexWithEnv(extraEnv = {}) {
  const env = { ...process.env, ...extraEnv };
  return run('npm', ['install', '-g', '@openai/codex@latest'], { env });
}

function main() {
  if (codexAvailable()) {
    return;
  }

  // First attempt: default global prefix
  if (!installGlobalCodexWithEnv()) {
    // Retry with user-level prefix without mutating global npm config
    const home = process.env.HOME || process.env.USERPROFILE;
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

