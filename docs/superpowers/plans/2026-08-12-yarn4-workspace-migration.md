# Yarn 4 鍓嶇宸ヤ綔鍖鸿縼绉诲疄鏂借鍒?

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 灏嗕袱涓?Vue 鍓嶇浠?pnpm workspace 杩佺Щ鍒?Yarn 4 鐨?`node_modules` linker锛屼慨澶?Ubuntu 涓婃棫 Yarn `fsevents`/peer 渚濊禆鍏煎闂銆?

**Architecture:** 鏍圭洰褰曚繚鐣欎竴涓?Yarn workspace锛屽伐浣滃尯涓?`frontend` 鍜?`admin-frontend`銆俌arn 4.9.2 閫氳繃 `.yarnrc.yml` 涓庨」鐩唴 release 鏂囦欢鍥哄畾锛屼娇鐢?`node_modules` linker锛涘墠绔緷璧栧拰涓氬姟浠ｇ爜涓嶆敼锛屽彧杩佺Щ鍖呯鐞嗗叆鍙ｃ€侀攣鏂囦欢銆佸绾︽祴璇曞拰閮ㄧ讲鏂囨。銆?

**Tech Stack:** Node.js >=20.19, Yarn 4.9.2, Vue 3, Vite 7, TypeScript 5, Vitest 4, Ubuntu + Corepack銆?

## Global Constraints

- 浣跨敤 Yarn锛屼笉鍐嶄娇鐢?pnpm 绠＄悊鍓嶇銆?
- 鍥哄畾 Yarn 4.9.2 鍜?`nodeLinker: node-modules`銆?
- 涓嶄慨鏀瑰墠绔笟鍔￠€昏緫銆?
- 涓嶆彁浜?`node_modules/`銆乣dist/`銆乣.env` 鎴?Yarn cache銆?
- Ubuntu 鏋勫缓蹇呴』鏀寔 `/admin/` 瀛愯矾寰勩€?

---

### Task 1: 鍥哄畾 Yarn 4 workspace 鍩虹閰嶇疆

**Files:**
- Create: `.yarnrc.yml`
- Create: `.yarn/releases/yarn-4.9.2.cjs`
- Modify: `package.json`
- Delete: `pnpm-workspace.yaml`

- [ ] **Step 1: Write the failing contract test**

鏇存柊 `tools/tests/test_frontend_yarn_contract.py` 涓?Yarn 濂戠害娴嬭瘯锛屾柇瑷€鏍规竻鍗曞寘鍚?`packageManager: yarn@4.9.2`锛宍.yarnrc.yml` 浣跨敤 `nodeLinker: node-modules`锛屼袱涓伐浣滃尯瀛樺湪锛屽苟涓?pnpm 閰嶇疆涓嶅瓨鍦ㄣ€?

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tools/tests/test_frontend_yarn_contract.py -q`
Expected: FAIL锛屽洜涓哄綋鍓嶄粨搴撲粛澹版槑 pnpm 骞跺瓨鍦?pnpm workspace 鏂囦欢銆?

- [ ] **Step 3: Implement minimal configuration**

灏嗘牴鑴氭湰鏀逛负锛?

```json
"build": "yarn workspaces foreach run build",
"test": "yarn workspaces foreach run test:run"
```

鏂板锛?

```yaml
nodeLinker: node-modules
yarnPath: .yarn/releases/yarn-4.9.2.cjs
packageExtensions:
  "@floating-ui/vue@*":
    peerDependencies:
      vue: "*"
```

鐢?Corepack 鑾峰彇 Yarn 4.9.2 release 鏂囦欢锛屽垹闄?pnpm workspace 閰嶇疆銆?

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tools/tests/test_frontend_yarn_contract.py -q`
Expected: PASS銆?

- [ ] **Step 5: Commit**

```bash
git add package.json .yarnrc.yml .yarn pnpm-workspace.yaml tools/tests/test_frontend_yarn_contract.py
git commit -m "chore: configure yarn 4 frontend workspace"
```

### Task 2: 鐢熸垚 Yarn 閿佹枃浠跺苟淇 peer 渚濊禆濂戠害

**Files:**
- Create: `yarn.lock`
- Delete: `pnpm-lock.yaml`
- Modify: `frontend/package.json`
- Modify: `admin-frontend/package.json`
- Modify: `tools/tests/test_version_contract.py`

- [ ] **Step 1: Add failing peer contract assertions**

鍦ㄥ寘绠＄悊鍣ㄥ绾︽祴璇曚腑鏂█涓や釜鍓嶇 workspace 鏄惧紡鎻愪緵 `vue`锛屽苟鏂█ Yarn lock 鏂囦欢瀛樺湪銆乸npm lock 鏂囦欢涓嶅瓨鍦ㄣ€?

- [ ] **Step 2: Run focused tests to verify RED**

Run: `python -m pytest tools/tests/test_frontend_yarn_contract.py tools/tests/test_version_contract.py -q`
Expected: FAIL锛屽洜涓哄綋鍓嶄粛浣跨敤 pnpm lock锛屼笖 root peer 渚涚粰灏氭湭澹版槑銆?

- [ ] **Step 3: Implement lock and peer declarations**

鍦ㄤ袱涓墠绔殑 `devDependencies` 澧炲姞涓庡綋鍓?Vue 鍏煎鐨?`vue` peer 渚涚粰澹版槑锛堜繚鎸佺幇鏈?`dependencies.vue` 涓嶅彉锛夛紝鎵ц `corepack yarn install` 鐢熸垚 Yarn 4 lock 鏂囦欢锛岀‘璁?`fsevents` 鍦?Linux 涓?optional Darwin-only 鍖呫€?

- [ ] **Step 4: Run focused tests and dependency inspection**

Run:

```bash
python -m pytest tools/tests/test_frontend_yarn_contract.py tools/tests/test_version_contract.py -q
corepack yarn why vue
corepack yarn why vue-demi
```

Expected: PASS锛屼笖渚濊禆閾捐В鏋愬埌鍗曚竴 Vue 鐗堟湰銆?

- [ ] **Step 5: Commit**

```bash
git add yarn.lock frontend/package.json admin-frontend/package.json pnpm-lock.yaml tools/tests/test_frontend_yarn_contract.py tools/tests/test_version_contract.py
git commit -m "chore: migrate frontend dependencies to yarn lockfile"
```

### Task 3: 杩佺Щ鑴氭湰銆侀儴缃叉枃妗ｄ笌 smoke 鍏ュ彛

**Files:**
- Modify: `README.md`
- Modify: `deploy/ubuntu/ADMIN_FRONTEND_SUBPATH.md`
- Modify: `docs/offline-platform-runbook.md`
- Modify: `tools/start_smoke_frontend.cmd`
- Modify: `tools/start_smoke_admin.cmd`
- Modify: `tools/tests/test_frontend_yarn_contract.py`

- [ ] **Step 1: Extend failing command contract tests**

鏂█ README銆侀儴缃叉枃妗ｅ拰涓や釜 smoke 鑴氭湰鍙娇鐢?Yarn 鍛戒护锛屼笉鑳藉嚭鐜?pnpm 鏋勫缓/鍚姩鍛戒护銆?

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tools/tests/test_frontend_yarn_contract.py -q`
Expected: FAIL锛屽洜涓哄綋鍓嶆枃妗ｅ拰 smoke 鑴氭湰浠嶄娇鐢?pnpm銆?

- [ ] **Step 3: Replace commands**

浣跨敤浠ヤ笅鍛戒护鏍煎紡锛?

```bash
corepack yarn install --immutable
corepack yarn workspace dc-agent-frontend dev
corepack yarn workspace dc-agent-admin-frontend dev
corepack yarn workspace dc-agent-frontend build
VITE_ADMIN_BASE_PATH=/admin/ corepack yarn workspace dc-agent-admin-frontend build
```

Windows smoke 鑴氭湰浣跨敤 `corepack yarn.cmd workspace ...`锛屼笉浣跨敤 pnpm 鎴?npm銆?

- [ ] **Step 4: Run command contract tests**

Run: `python -m pytest tools/tests/test_frontend_yarn_contract.py -q`
Expected: PASS銆?

- [ ] **Step 5: Commit**

```bash
git add README.md deploy/ubuntu/ADMIN_FRONTEND_SUBPATH.md docs/offline-platform-runbook.md tools/start_smoke_frontend.cmd tools/start_smoke_admin.cmd tools/tests/test_frontend_yarn_contract.py
git commit -m "docs: switch frontend deployment commands to yarn"
```

### Task 4: 瀹屾暣瀹夎銆佹瀯寤哄拰娴嬭瘯楠岃瘉

**Files:** None (verification only)

- [ ] **Step 1: Clean generated dependencies**

```bash
rm -rf node_modules frontend/node_modules admin-frontend/node_modules
```

- [ ] **Step 2: Install from immutable Yarn lock**

```bash
corepack yarn install --immutable
```

- [ ] **Step 3: Run all frontend tests**

```bash
corepack yarn workspaces foreach run test:run
```

- [ ] **Step 4: Build user frontend and admin subpath frontend**

```bash
corepack yarn workspace dc-agent-frontend build
VITE_ADMIN_BASE_PATH=/admin/ corepack yarn workspace dc-agent-admin-frontend build
```

- [ ] **Step 5: Run repository contracts and diff checks**

```bash
python -m pytest tools/tests/test_frontend_yarn_contract.py tools/tests/test_version_contract.py -q
git diff --check
git status --short
```

- [ ] **Step 6: Commit final verification adjustments if needed**

Only commit if a test or documentation correction was required; do not commit generated `node_modules` or `dist` output.
