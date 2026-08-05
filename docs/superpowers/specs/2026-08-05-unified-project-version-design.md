# 统一项目版本设计

## 目标

DC-Agent 的后端、用户端和管理端分别使用独立语义化版本，并在对应应用代码修改完成后默认递增该应用的 patch 版本。

## 设计

后端 `backend/app/__init__.py` 中的 `__version__` 是后端版本源，PDM Backend 和 FastAPI 使用该值。用户端和管理端分别以自己的 `package.json` 为版本源。`tools/bump_version.py` 接受 `backend`、`frontend` 或 `admin-frontend` 组件参数，只更新选中应用及其对应锁文件。

默认发布流程执行 `python tools/bump_version.py <component> patch`。兼容性功能使用 `minor`，不兼容变更使用 `major`，也可以传入明确版本号。自动测试确认升级一个应用不会改动其他应用。

## 错误处理与验证

版本工具只接受 `X.Y.Z` 或 `patch/minor/major`，文件结构不符合预期时立即失败，不写入猜测结果。提交前运行版本契约测试、Ruff、后端测试以及两个前端的测试和构建。
