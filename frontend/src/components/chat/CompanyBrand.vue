<template>
  <header class="company-brand" data-ignore-quantum-pulse aria-label="公司品牌">
    <img class="company-brand__mark" :src="'/favicon-logo.svg'" alt="" aria-hidden="true">
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
  display: block;
  width: 38px;
  height: 38px;
  flex: 0 0 auto;
  object-fit: contain;
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
  }

  .company-brand__name {
    max-width: 150px;
    font-size: 14px;
  }
}
</style>
