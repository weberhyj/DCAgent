from __future__ import annotations

from pathlib import Path
from time import sleep

import httpx
from PIL import Image, ImageChops, ImageStat
from playwright.sync_api import Page, expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = ROOT / "qa-screenshots"
BACKEND_URL = "http://127.0.0.1:8015"
USER_URL = "http://127.0.0.1:5177"
ADMIN_URL = "http://127.0.0.1:5178"


def assert_image_has_detail(path: Path, min_colors: int = 32) -> None:
    image = Image.open(path).convert("RGB").resize((160, 90))
    colors = image.getcolors(maxcolors=160 * 90)
    color_count = 0 if colors is None else len(colors)
    if color_count < min_colors:
        raise AssertionError(f"{path.name} looks blank; only {color_count} colors found")


def assert_images_changed(before: Path, after: Path, min_mean_delta: float = 0.05) -> None:
    first = Image.open(before).convert("RGB").resize((160, 90))
    second = Image.open(after).convert("RGB").resize((160, 90))
    diff = ImageChops.difference(first, second)
    mean_delta = sum(ImageStat.Stat(diff).mean) / 3
    if mean_delta < min_mean_delta:
        raise AssertionError(
            f"{before.name} and {after.name} are too similar; mean delta {mean_delta:.4f}",
        )


def watch_errors(page: Page, label: str) -> list[str]:
    errors: list[str] = []

    def record_console(message) -> None:
        if message.type == "error":
            errors.append(f"{label} console error: {message.text}")

    def record_page_error(error: Exception) -> None:
        errors.append(f"{label} page error: {error}")

    page.on("console", record_console)
    page.on("pageerror", record_page_error)
    return errors


def seed_knowledge_base() -> None:
    content = (
        "差旅报销资料要求：员工提交差旅票据材料时，必须提供发票、行程单、"
        "审批记录和费用说明。票据缺失时，财务可以退回补充。"
    )
    with httpx.Client(timeout=10.0) as client:
        response = client.post(
            f"{BACKEND_URL}/api/knowledge/uploads",
            data={"classification": "内部·机密"},
            files={"files": ("travel-policy.txt", content.encode("utf-8"), "text/plain")},
        )
        response.raise_for_status()

        for _ in range(20):
            sources = client.get(f"{BACKEND_URL}/api/knowledge/sources").json()
            if any(source["status"] == "已索引" and source["records"] > 0 for source in sources):
                return
            sleep(0.2)

    raise AssertionError("Seeded knowledge source was not indexed in time")


def verify_quantum_canvas(page: Page, prefix: str) -> None:
    canvas = page.locator("canvas").first
    expect(canvas).to_be_visible(timeout=20_000)
    box = canvas.bounding_box()
    if box is None:
        raise AssertionError(f"{prefix} canvas has no bounding box")
    viewport = page.viewport_size or {"width": 0, "height": 0}
    if box["width"] < viewport["width"] * 0.9 or box["height"] < viewport["height"] * 0.9:
        raise AssertionError(f"{prefix} canvas does not cover the viewport: {box}")

    before = SCREENSHOT_DIR / f"runtime-{prefix}-canvas-before.png"
    after = SCREENSHOT_DIR / f"runtime-{prefix}-canvas-after.png"
    canvas.screenshot(path=str(before))
    page.wait_for_timeout(900)
    canvas.screenshot(path=str(after))
    assert_image_has_detail(before)
    assert_image_has_detail(after)
    assert_images_changed(before, after)


def verify_user_app() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for prefix, viewport in (
            ("user-desktop", {"width": 1440, "height": 900}),
            ("user-mobile", {"width": 390, "height": 844}),
        ):
            page = browser.new_page(viewport=viewport)
            errors = watch_errors(page, prefix)
            page.goto(USER_URL, wait_until="networkidle")
            expect(page.locator(".knowledge-search-hero")).to_be_visible(timeout=20_000)
            expect(page.locator(".split-text-title")).to_have_attribute(
                "aria-label",
                "欢迎来到DC智识中枢",
            )
            verify_quantum_canvas(page, prefix)
            page.screenshot(path=str(SCREENSHOT_DIR / f"runtime-{prefix}.png"), full_page=True)
            if errors:
                raise AssertionError("\n".join(errors))
        page.close()

        seed_knowledge_base()

        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = watch_errors(page, "user-flow")
        page.goto(USER_URL, wait_until="networkidle")
        page.locator('input[type="text"]').first.fill("差旅票据材料需要什么")
        page.locator('button[type="submit"]').first.click()
        launch_loader = page.locator('[data-testid="knowledge-launch-loader"]')
        expect(launch_loader).to_be_visible(timeout=4_000)
        expect(launch_loader).to_have_attribute("data-ignore-quantum-pulse", "")
        expect(page.locator(".answer-panel")).to_be_visible(timeout=8_000)
        expect(page.locator(".query-header h1")).to_have_text("DC智识中枢")
        expect(page.locator(".message.user")).to_have_count(1)
        expect(page.locator(".message.assistant")).to_have_count(1)
        answer = page.locator(".answer-paragraph").first
        expect(answer).to_contain_text("发票", timeout=10_000)
        expect(answer).to_contain_text("行程单")
        if "未检索到足够依据" in answer.text_content():
            raise AssertionError("User flow fell back to no-evidence answer after seeding knowledge")
        page.screenshot(path=str(SCREENSHOT_DIR / "runtime-user-answer-panel.png"), full_page=True)
        if errors:
            raise AssertionError("\n".join(errors))
        page.close()
        browser.close()


def verify_admin_app() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors = watch_errors(page, "admin")
        page.goto(ADMIN_URL, wait_until="networkidle")
        expect(page.locator('[data-testid="overview-page"]')).to_be_visible(timeout=20_000)
        expect(page.get_by_text("管理概览", exact=True).first).to_be_visible()
        page.screenshot(path=str(SCREENSHOT_DIR / "runtime-admin-overview.png"), full_page=True)

        page.goto(f"{ADMIN_URL}/knowledge", wait_until="networkidle")
        expect(page.locator('[data-testid="knowledge-management-page"]')).to_be_visible(timeout=20_000)
        expect(page.get_by_text("资料投喂")).to_be_visible()
        expect(page.get_by_text("已接入资料")).to_be_visible()
        page.screenshot(path=str(SCREENSHOT_DIR / "runtime-admin-knowledge.png"), full_page=True)
        page.locator('[data-testid="open-knowledge-upload"]').click()
        expect(page.locator('[data-testid="knowledge-upload-form"]')).to_be_visible()
        expect(page.get_by_role("dialog", name="资料投喂")).to_be_visible()
        page.screenshot(path=str(SCREENSHOT_DIR / "runtime-admin-upload-dialog.png"), full_page=True)
        page.locator('[data-testid="base-dialog-close"]').click()
        expect(page.locator('[data-testid="knowledge-upload-form"]')).to_be_hidden()

        with httpx.Client(timeout=10.0) as client:
            sources = client.get(f"{BACKEND_URL}/api/knowledge/sources").json()
        source_id = sources[0]["id"]
        page.goto(f"{ADMIN_URL}/knowledge/{source_id}", wait_until="networkidle")
        expect(page.locator('[data-testid="knowledge-source-detail-page"]')).to_be_visible(timeout=20_000)
        expect(page.locator(".chunk-panel > header strong")).to_have_text("解析片段")
        page.screenshot(path=str(SCREENSHOT_DIR / "runtime-admin-source-detail.png"), full_page=True)

        page.goto(f"{ADMIN_URL}/agent-runs", wait_until="networkidle")
        expect(page.locator('[data-testid="agent-audit-page"]')).to_be_visible(timeout=20_000)
        expect(page.get_by_role("heading", name="Agent 执行审计", level=1)).to_be_visible()
        expect(page.get_by_text("差旅票据材料需要什么")).to_be_visible()
        expect(page.get_by_text("检索知识库").first).to_be_visible()
        page.screenshot(path=str(SCREENSHOT_DIR / "runtime-admin-agent-audit.png"), full_page=True)
        if errors:
            raise AssertionError("\n".join(errors))
        page.close()

        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile_errors = watch_errors(mobile, "admin-mobile")
        mobile.goto(f"{ADMIN_URL}/overview", wait_until="networkidle")
        expect(mobile.locator('[data-testid="overview-page"]')).to_be_visible(timeout=20_000)
        expect(mobile.locator(".admin-nav")).to_be_visible()
        mobile.screenshot(path=str(SCREENSHOT_DIR / "runtime-admin-mobile-overview.png"), full_page=True)
        mobile.goto(f"{ADMIN_URL}/knowledge", wait_until="networkidle")
        expect(mobile.locator('[data-testid="knowledge-management-page"]')).to_be_visible(timeout=20_000)
        mobile.screenshot(path=str(SCREENSHOT_DIR / "runtime-admin-mobile-knowledge.png"), full_page=True)
        if mobile_errors:
            raise AssertionError("\n".join(mobile_errors))
        mobile.close()
        browser.close()


def main() -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    verify_user_app()
    verify_admin_app()
    print("UI smoke passed")


if __name__ == "__main__":
    main()
