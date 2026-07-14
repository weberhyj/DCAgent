<template>
  <header class="company-brand" data-ignore-quantum-pulse aria-label="公司品牌">
    <span class="company-brand__mark" aria-hidden="true">DC</span>
    <span class="company-brand__name">DC-Agent</span>
  </header>
</template>

<style scoped>
.company-brand {
  position: absolute;
  top: 22px;
  left: 26px;
  z-index: 4;
  display: inline-flex;
  align-items: center;
  gap: 11px;
  min-width: 0;
  color: #edfaff;
}

.company-brand__mark {
  display: inline-grid;
  place-items: center;
  width: 38px;
  height: 38px;
  border: 1px solid rgba(129, 229, 255, 0.32);
  border-radius: 12px;
  color: #dff9ff;
  background:
    linear-gradient(145deg, rgba(184, 242, 255, 0.2), rgba(103, 216, 255, 0.06)),
    rgba(10, 20, 28, 0.48);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.1),
    0 0 26px rgba(80, 198, 238, 0.16);
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  backdrop-filter: blur(14px);
}

.company-brand__name {
  overflow: hidden;
  max-width: 220px;
  font-size: 15px;
  font-weight: 650;
  letter-spacing: 0;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-shadow: 0 0 18px rgba(103, 216, 255, 0.18);
}

@media (max-width: 920px) {
  .company-brand {
    top: 14px;
    left: 14px;
  }

  .company-brand__mark {
    width: 34px;
    height: 34px;
    border-radius: 11px;
  }

  .company-brand__name {
    max-width: 150px;
    font-size: 14px;
  }
}
</style>
