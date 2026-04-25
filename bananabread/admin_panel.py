ADMIN_PANEL_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BananaBread Management</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #10100d;
      --panel: #191812;
      --panel-2: #232116;
      --ink: #f7edcf;
      --muted: #b9ad8b;
      --line: #3b3522;
      --gold: #f0c75e;
      --green: #85d58a;
      --red: #ff7f66;
      --shadow: 0 22px 70px rgba(0, 0, 0, .38);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      color: var(--ink);
      background:
        radial-gradient(circle at 18% 12%, rgba(240, 199, 94, .22), transparent 28rem),
        radial-gradient(circle at 88% 4%, rgba(133, 213, 138, .12), transparent 24rem),
        linear-gradient(135deg, #0d0d0a, var(--bg) 48%, #17130b);
    }
    main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 44px 0; }
    header { display: grid; gap: 14px; margin-bottom: 28px; }
    h1 { font-family: Georgia, 'Times New Roman', serif; font-size: clamp(2.4rem, 6vw, 5.8rem); line-height: .86; margin: 0; letter-spacing: -.07em; }
    h2 { margin: 0 0 16px; font-size: 1rem; color: var(--gold); text-transform: uppercase; letter-spacing: .14em; }
    p { color: var(--muted); line-height: 1.6; margin: 0; max-width: 780px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
    .card { background: linear-gradient(180deg, rgba(35, 33, 22, .94), rgba(25, 24, 18, .94)); border: 1px solid var(--line); border-radius: 24px; padding: 22px; box-shadow: var(--shadow); }
    .wide { grid-column: 1 / -1; }
    label { display: block; color: var(--muted); font-size: .78rem; margin: 12px 0 7px; text-transform: uppercase; letter-spacing: .08em; }
    input, textarea, select {
      width: 100%; border: 1px solid var(--line); border-radius: 14px; background: #0d0d0a; color: var(--ink); padding: 12px 13px; font: inherit; outline: none;
    }
    textarea { min-height: 150px; resize: vertical; }
    input:focus, textarea:focus, select:focus { border-color: var(--gold); box-shadow: 0 0 0 3px rgba(240, 199, 94, .16); }
    button { border: 0; border-radius: 999px; margin-top: 16px; padding: 12px 18px; font: inherit; font-weight: 700; color: #18140a; background: var(--gold); cursor: pointer; }
    button.secondary { color: var(--ink); background: transparent; border: 1px solid var(--line); }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .status { margin: 0 0 18px; padding: 13px 15px; border: 1px solid var(--line); border-radius: 16px; color: var(--muted); background: rgba(0,0,0,.22); }
    .ok { color: var(--green); }
    .bad { color: var(--red); }
    .users { display: grid; gap: 10px; }
    .user { display: grid; grid-template-columns: 1fr auto; gap: 10px; padding: 13px; border: 1px solid var(--line); border-radius: 16px; background: rgba(0,0,0,.18); }
    .meta { color: var(--muted); font-size: .82rem; margin-top: 6px; }
    code { color: var(--gold); }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 12px 0 0; padding: 14px; border-radius: 14px; background: #0d0d0a; color: var(--green); }
    .header-bar { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 12px; }
    @media (max-width: 820px) { .grid, .row { grid-template-columns: 1fr; } main { width: min(100% - 22px, 1180px); padding: 24px 0; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="header-bar">
        <h1>Tenant Control Room</h1>
        <button class="secondary" id="logoutBtn" style="margin-top:0;">Sign out</button>
      </div>
      <p>Configure BananaBread management access, default token limits, tiers, user API keys, and cache isolation.</p>
    </header>

    <div id="status" class="status">Loading configuration...</div>

    <section class="grid">
      <div class="card">
        <h2>Global Defaults</h2>
        <div class="row">
          <div><label for="dailyLimit">Daily tokens</label><input id="dailyLimit" type="number" min="0" placeholder="Unlimited"></div>
          <div><label for="weeklyLimit">Weekly tokens</label><input id="weeklyLimit" type="number" min="0" placeholder="Unlimited"></div>
        </div>
        <label for="newManagementKey">Rotate management key</label>
        <input id="newManagementKey" type="password" autocomplete="new-password" placeholder="Leave blank to keep current key">
        <button id="saveConfig">Save global settings</button>
      </div>

      <div class="card">
        <h2>Cache</h2>
        <p>Cache can be <strong>global</strong> (shared across users) or <strong>per-user</strong> (isolated). Sizes are in MB. Leave blank to use the server default.</p>
        <label for="cacheScope">Cache scope</label>
        <select id="cacheScope">
          <option value="global">Global (shared across users)</option>
          <option value="per_user">Per-user (isolated)</option>
        </select>
        <div class="row">
          <div><label for="defaultEmbedMb">Default embedding cache MB</label><input id="defaultEmbedMb" type="number" min="0" placeholder="Server default"></div>
          <div><label for="defaultRerankMb">Default rerank cache MB</label><input id="defaultRerankMb" type="number" min="0" placeholder="Server default"></div>
        </div>
        <button id="saveCache">Save cache settings</button>
      </div>

      <div class="card">
        <h2>Tiers</h2>
        <p>JSON object where each tier has optional <code>daily</code> and <code>weekly</code> limits.</p>
        <label for="tiersJson">Tier limits</label>
        <textarea id="tiersJson" spellcheck="false" placeholder='{"free":{"daily":10000,"weekly":50000}}'></textarea>
        <button id="saveTiers">Save tiers</button>
      </div>

      <div class="card">
        <h2>Create User</h2>
        <label for="username">Username</label>
        <input id="username" placeholder="alice">
        <label for="userTier">Tier</label>
        <select id="userTier"><option value="">No tier</option></select>
        <div class="row">
          <div><label for="userDaily">Daily override</label><input id="userDaily" type="number" min="0" placeholder="Tier/default"></div>
          <div><label for="userWeekly">Weekly override</label><input id="userWeekly" type="number" min="0" placeholder="Tier/default"></div>
        </div>
        <button id="createUser">Create user and key</button>
        <pre id="createdKey" hidden></pre>
      </div>

      <div class="card wide">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
          <h2 style="margin:0;">Users</h2>
          <button class="secondary" id="regenerateAllBtn" style="margin-top:0;">Regenerate All</button>
        </div>
        <div id="users" class="users"><p>Loading...</p></div>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const status = $('status');
    let currentConfig = null;

    function setStatus(message, ok = true) {
      status.textContent = message;
      status.className = `status ${ok ? 'ok' : 'bad'}`;
    }

    function parseLimit(value) {
      return value === '' ? null : Number(value);
    }

    function apiHeaders() {
      return { 'Content-Type': 'application/json' };
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        credentials: 'same-origin',
        headers: { ...apiHeaders(), ...(options.headers || {}) }
      });
      if (response.status === 401 || response.status === 403) {
        window.location.href = '/management/login';
        throw new Error('Session expired. Redirecting to login...');
      }
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || body));
      return body;
    }

    function render(config) {
      currentConfig = config;
      $('dailyLimit').value = config.default_limits.daily ?? '';
      $('weeklyLimit').value = config.default_limits.weekly ?? '';
      $('tiersJson').value = JSON.stringify(config.tiers || {}, null, 2);

      const cache = config.cache || {};
      $('cacheScope').value = cache.scope || 'global';
      $('defaultEmbedMb').value = cache.default_embedding_mb ?? '';
      $('defaultRerankMb').value = cache.default_rerank_mb ?? '';

      const tierSelect = $('userTier');
      tierSelect.innerHTML = '<option value="">No tier</option>';
      Object.keys(config.tiers || {}).forEach((tier) => {
        const option = document.createElement('option');
        option.value = tier;
        option.textContent = tier;
        tierSelect.appendChild(option);
      });

      const users = Object.entries(config.users || {});
      $('users').innerHTML = users.length ? '' : '<p>No users yet.</p>';
      users.forEach(([name, user]) => {
        const daily = user.usage?.daily;
        const weekly = user.usage?.weekly;
        const node = document.createElement('div');
        node.className = 'user';
        node.innerHTML = `<div><strong>${name}</strong><div class="meta">key ${user.api_key_preview || 'none'} · tier ${user.tier || 'none'} · daily ${daily?.tokens ?? 0}/${user.effective_limits.daily ?? '∞'} · weekly ${weekly?.tokens ?? 0}/${user.effective_limits.weekly ?? '∞'}</div></div><div style="display:flex;gap:8px;align-items:center;"><code>${user.api_key_preview || ''}</code><button class="secondary regenerate-btn" data-user="${name}" style="margin-top:0;padding:6px 12px;font-size:.8rem;">Regenerate</button></div>`;
        $('users').appendChild(node);
      });

      document.querySelectorAll('.regenerate-btn').forEach((btn) => {
        btn.onclick = async () => {
          const user = btn.dataset.user;
          if (!confirm(`Regenerate API key for ${user}? The old key will stop working immediately.`)) return;
          try {
            const result = await api(`/v1/management/users/${encodeURIComponent(user)}/regenerate`, { method: 'POST' });
            setStatus(`Regenerated key for ${result.username}.`);
            $('createdKey').hidden = false;
            $('createdKey').textContent = `Regenerated ${result.username}\\nNew API key: ${result.api_key}`;
            await loadConfig();
          } catch (error) { setStatus(error.message, false); }
        };
      });
    }

    async function loadConfig() {
      const config = await api('/v1/management/config');
      render(config);
      setStatus('Configuration loaded.');
    }

    $('logoutBtn').onclick = () => {
      window.location.href = '/management/logout';
    };

    $('saveConfig').onclick = async () => {
      try {
        const payload = { default_limits: { daily: parseLimit($('dailyLimit').value), weekly: parseLimit($('weeklyLimit').value) } };
        const newKey = $('newManagementKey').value.trim();
        if (newKey) payload.management_key = newKey;
        const config = await api('/v1/management/config', { method: 'PATCH', body: JSON.stringify(payload) });
        if (newKey) $('newManagementKey').value = '';
        render(config);
        setStatus('Global settings saved.');
      } catch (error) { setStatus(error.message, false); }
    };

    $('saveCache').onclick = async () => {
      try {
        const payload = {
          cache: {
            scope: $('cacheScope').value,
            default_embedding_mb: parseLimit($('defaultEmbedMb').value),
            default_rerank_mb: parseLimit($('defaultRerankMb').value),
            users: {}
          }
        };
        const config = await api('/v1/management/config', { method: 'PATCH', body: JSON.stringify(payload) });
        render(config);
        setStatus('Cache settings saved.');
      } catch (error) { setStatus(error.message, false); }
    };

    $('saveTiers').onclick = async () => {
      try {
        const tiers = JSON.parse($('tiersJson').value || '{}');
        const config = await api('/v1/management/config', { method: 'PATCH', body: JSON.stringify({ tiers }) });
        render(config);
        setStatus('Tiers saved.');
      } catch (error) { setStatus(error.message, false); }
    };

    $('createUser').onclick = async () => {
      try {
        const limits = {};
        if ($('userDaily').value !== '') limits.daily = parseLimit($('userDaily').value);
        if ($('userWeekly').value !== '') limits.weekly = parseLimit($('userWeekly').value);
        const created = await api('/v1/management/users', {
          method: 'POST',
          body: JSON.stringify({ username: $('username').value, tier: $('userTier').value || null, limits: Object.keys(limits).length ? limits : null })
        });
        $('createdKey').hidden = false;
        $('createdKey').textContent = `Created ${created.username}\\nAPI key: ${created.api_key}`;
        $('username').value = '';
        $('userDaily').value = '';
        $('userWeekly').value = '';
        await loadConfig();
      } catch (error) { setStatus(error.message, false); }
    };

    $('regenerateAllBtn').onclick = async () => {
      if (!currentConfig) return;
      const users = Object.keys(currentConfig.users || {});
      if (!users.length) { setStatus('No users to regenerate.', false); return; }
      if (!confirm(`Regenerate API keys for all ${users.length} user(s)? The old keys will stop working immediately.`)) return;
      try {
        const result = await api('/v1/management/users/regenerate', {
          method: 'POST',
          body: JSON.stringify({ usernames: users })
        });
        const okCount = result.regenerated?.length || 0;
        const errCount = result.errors?.length || 0;
        let msg = `Regenerated ${okCount} key(s).`;
        if (errCount) msg += ` ${errCount} failed.`;
        setStatus(msg);
        if (okCount) {
          $('createdKey').hidden = false;
          $('createdKey').textContent = result.regenerated.map((r) => `${r.username}: ${r.api_key}`).join('\\n');
        }
        await loadConfig();
      } catch (error) { setStatus(error.message, false); }
    };

    loadConfig().catch((error) => setStatus(error.message, false));
  </script>
</body>
</html>"""

LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BananaBread Management - Login</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #10100d;
      --panel: #191812;
      --panel-2: #232116;
      --ink: #f7edcf;
      --muted: #b9ad8b;
      --line: #3b3522;
      --gold: #f0c75e;
      --green: #85d58a;
      --red: #ff7f66;
      --shadow: 0 22px 70px rgba(0, 0, 0, .38);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      color: var(--ink);
      background:
        radial-gradient(circle at 18% 12%, rgba(240, 199, 94, .22), transparent 28rem),
        radial-gradient(circle at 88% 4%, rgba(133, 213, 138, .12), transparent 24rem),
        linear-gradient(135deg, #0d0d0a, var(--bg) 48%, #17130b);
    }
    .login-box {
      background: linear-gradient(180deg, rgba(35, 33, 22, .94), rgba(25, 24, 18, .94));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 36px;
      width: min(420px, calc(100% - 32px));
      box-shadow: var(--shadow);
    }
    h1 { font-family: Georgia, 'Times New Roman', serif; font-size: 2rem; margin: 0 0 8px; letter-spacing: -.04em; }
    p { color: var(--muted); margin: 0 0 24px; line-height: 1.5; }
    label { display: block; color: var(--muted); font-size: .78rem; margin: 16px 0 7px; text-transform: uppercase; letter-spacing: .08em; }
    input {
      width: 100%; border: 1px solid var(--line); border-radius: 14px; background: #0d0d0a; color: var(--ink); padding: 12px 13px; font: inherit; outline: none;
    }
    input:focus { border-color: var(--gold); box-shadow: 0 0 0 3px rgba(240, 199, 94, .16); }
    button { width: 100%; border: 0; border-radius: 999px; margin-top: 20px; padding: 12px 18px; font: inherit; font-weight: 700; color: #18140a; background: var(--gold); cursor: pointer; }
    .status { margin-top: 14px; padding: 10px 14px; border-radius: 12px; font-size: .9rem; display: none; }
    .status.bad { display: block; background: rgba(255, 127, 102, .12); color: var(--red); border: 1px solid rgba(255, 127, 102, .25); }
    .status.ok { display: block; background: rgba(133, 213, 138, .12); color: var(--green); border: 1px solid rgba(133, 213, 138, .25); }
  </style>
</head>
<body>
  <div class="login-box">
    <h1>Management</h1>
    <p>Enter your management key to access the tenant control room.</p>
    <form id="loginForm">
      <label for="key">Management key</label>
      <input id="key" type="password" autocomplete="current-password" placeholder="Bearer token" required autofocus>
      <button type="submit">Sign in</button>
      <div id="status" class="status"></div>
    </form>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    $('loginForm').onsubmit = async (e) => {
      e.preventDefault();
      const status = $('status');
      status.className = 'status';
      try {
        const response = await fetch('/management/auth', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ key: $('key').value.trim() }),
          credentials: 'same-origin'
        });
        if (response.ok) {
          window.location.href = '/management';
        } else {
          const body = await response.json().catch(() => ({}));
          status.textContent = body.detail || 'Authentication failed';
          status.className = 'status bad';
        }
      } catch (err) {
        status.textContent = 'Network error: ' + err.message;
        status.className = 'status bad';
      }
    };
  </script>
</body>
</html>"""
