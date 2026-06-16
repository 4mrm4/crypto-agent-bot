"""Take screenshot of the UI and save for README."""
from playwright.sync_api import sync_playwright
import os

html_path = r"C:\Trading-bot\crypto_agent_bot\ui\index.html"
screenshot_path = r"C:\Trading-bot\crypto_agent_bot\docs\ui-screenshot.png"
file_url = "file:///" + html_path.replace("\\", "/")

os.makedirs(r"C:\Trading-bot\crypto_agent_bot\docs", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(file_url, wait_until="networkidle")
    page.wait_for_timeout(2000)  # let React render + animations settle
    page.screenshot(path=screenshot_path, full_page=False)
    browser.close()

print(f"Screenshot saved: {screenshot_path}")
print(f"Size: {os.path.getsize(screenshot_path)} bytes")
