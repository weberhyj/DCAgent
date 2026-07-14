import { createRouter, createWebHistory } from 'vue-router'
import AdminLayout from '@/components/layout/AdminLayout.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      component: AdminLayout,
      children: [
        {
          path: '',
          redirect: { name: 'overview' },
        },
        {
          path: 'overview',
          name: 'overview',
          component: () => import('@/views/OverviewPage.vue'),
          meta: { title: '管理概览' },
        },
        {
          path: 'knowledge',
          name: 'knowledge',
          component: () => import('@/views/KnowledgeManagementPage.vue'),
          meta: { title: '知识库管理' },
        },
        {
          path: 'knowledge/:sourceId',
          name: 'knowledge-source-detail',
          component: () => import('@/views/KnowledgeSourceDetailPage.vue'),
          meta: { title: '资料解析详情' },
        },
        {
          path: 'agent-runs',
          name: 'agent-runs',
          component: () => import('@/views/AgentAuditPage.vue'),
          meta: { title: 'Agent 执行审计' },
        },
        {
          path: 'quality',
          component: () => import('@/views/QualityModuleLayout.vue'),
          children: [
            {
              path: '',
              redirect: { name: 'quality-cases' },
            },
            {
              path: 'cases',
              name: 'quality-cases',
              component: () => import('@/views/QualityCasesPage.vue'),
              meta: { title: '质量评测' },
            },
            {
              path: 'reports',
              name: 'quality-reports',
              component: () => import('@/views/QualityReportsPage.vue'),
              meta: { title: '评测报告' },
            },
            {
              path: 'reports/:batchId',
              name: 'quality-report-detail',
              component: () => import('@/views/QualityReportDetailPage.vue'),
              meta: { title: '评测报告详情' },
            },
          ],
        },
      ],
    },
  ],
})

export default router
