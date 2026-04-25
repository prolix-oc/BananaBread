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
    @media (max-width: 820px) { .grid, .row { grid-template-columns: 1fr; } main { width: min(100% - 22px, 1180px); padding: 24px 0; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Tenant Control Room</h1>
      <p>Configure BananaBread management access, default token limits, tiers, user API keys, and cache isolation. Every action below requires the management key in <code>api_keys.json</code>.</p>
    </header>

    <div id="status" class="status">Enter the management key and load configuration.</div>

    <section class="grid">
      <div class="card wide">
        <h2>Access</h2>
        <label for="managementKey">Current management key</label>
        <input id="managementKey" type="password" autocomplete="current-password" placeholder="Bearer token used for management calls">
        <button id="loadConfig">Load configuration</button>
        <button class="secondary" id="forgetKey">Forget saved key</button>
      </div>

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

      <div class="card">
        <h2>Users</h2>
        <div id="users" class="users"><p>No configuration loaded.</p></div>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const status = $('status');
    let currentConfig = null;

    $('managementKey').value = localStorage.getItem('bananabreadManagementKey') || '';

    function setStatus(message, ok = true) {
      status.textContent = message;
      status.className = `status ${ok ? 'ok' : 'bad'}`;
    }

    function parseLimit(value) {
      return value === '' ? null : Number(value);
    }

    function authHeaders() {
      const key = $('managementKey').value.trim();
      if (!key) throw new Error('Management key is required.');
      localStorage.setItem('bananabreadManagementKey', key);
      return { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' };
    }

    async function api(path, options = {}) {
      const response = await fetch(path, { ...options, headers: { ...authHeaders(), ...(options.headers || {}) } });
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
        node.innerHTML = `<div><strong>${name}</strong><div class="meta">key ${user.api_key_preview || 'none'} · tier ${user.tier || 'none'} · daily ${daily?.tokens ?? 0}/${user.effective_limits.daily ?? '∞'} · weekly ${weekly?.tokens ?? 0}/${user.effective_limits.weekly ?? '∞'}</div></div><code>${user.api_key_preview || ''}</code>`;
        $('users').appendChild(node);
      });
    }

    async function loadConfig() {
      const config = await api('/v1/management/config');
      render(config);
      setStatus('Configuration loaded.');
    }

    $('loadConfig').onclick = () => loadConfig().catch((error) => setStatus(error.message, false));
    $('forgetKey').onclick = () => { localStorage.removeItem('bananabreadManagementKey'); $('managementKey').value = ''; setStatus('Saved management key removed.'); };

    $('saveConfig').onclick = async () => {
      try {
        const payload = { default_limits: { daily: parseLimit($('dailyLimit').value), weekly: parseLimit($('weeklyLimit').value) } };
        const newKey = $('newManagementKey').value.trim();
        if (newKey) payload.management_key = newKey;
        const config = await api('/v1/management/config', { method: 'PATCH', body: JSON.stringify(payload) });
        if (newKey) { $('managementKey').value = newKey; localStorage.setItem('bananabreadManagementKey', newKey); $('newManagementKey').value = ''; }
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
        $('createdKey').textContent = `Created ${created.username}\nAPI key: ${created.api_key}`;
        $('username').value = '';
        $('userDaily').value = '';
        $('userWeekly').value = '';
        await loadConfig();
      } catch (error) { setStatus(error.message, false); }
    };
  </script>
</body>
</html>"""
