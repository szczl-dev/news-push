#!/usr/bin/env python3
"""
美国新闻每日推送系统
- 每天北京时间 8:00 执行
- 获取前一天 12:00 ~ 当天 6:00（北京时间）的美国新闻
- AI 总结美股、AI/NVIDIA 相关内容
- 推送到飞书群

Usage:
    python3 main.py                          # 正常模式
    python3 main.py --debug                  # 调试模式
    python3 main.py --test-feishu            # 仅测试飞书连通性
    python3 main.py --window 6-12            # 自定义时间窗口
"""

import os
import sys
import re
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict
from typing import Optional

import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).parent / ".env")

LOG = logging.getLogger("news-push")
LOG.setLevel(logging.INFO)
LOG.addHandler(logging.StreamHandler(sys.stdout))

# Timezone
BEIJING_TZ = timezone(timedelta(hours=8))

# Feishu
FEISHU_WEBHOOK = os.getenv(
    "FEISHU_WEBHOOK",
    "https://open.feishu.cn/open-apis/bot/v2/hook/b2098d0a-8142-4847-8ed9-38ac7c18ba5c",
)

# Anthropic (DeepSeek compatible)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro")

# HTTP Proxy (optional, for accessing blocked feeds)
HTTP_PROXY = os.getenv("HTTP_PROXY", "")
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")

# Time window (Beijing time): previous day START_HOUR → today END_HOUR
WINDOW_START_HOUR = int(os.getenv("NEWS_WINDOW_START_HOUR", "12"))
WINDOW_END_HOUR = int(os.getenv("NEWS_WINDOW_END_HOUR", "6"))

# Max articles to feed to LLM (to control token usage)
MAX_ARTICLES_FOR_LLM = int(os.getenv("MAX_ARTICLES_FOR_LLM", "150"))

# ---------------------------------------------------------------------------
# RSS Feeds (Tier 1 may be blocked; Tier 2/3 are more accessible)
# ---------------------------------------------------------------------------
RSS_FEEDS = [
    # === Tier 1: Bloomberg / WSJ / NYT / Fortune (may need proxy) ===
    {"url": "https://feeds.bloomberg.com/markets/news.rss", "name": "Bloomberg Markets", "tier": 1},
    {"url": "https://feeds.bloomberg.com/technology/news.rss", "name": "Bloomberg Tech", "tier": 1},
    {"url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "name": "WSJ Markets", "tier": 1},
    {"url": "https://feeds.a.dj.com/rss/RSSWSJD.xml", "name": "WSJ Tech", "tier": 1},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml", "name": "NYT Tech", "tier": 1},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "name": "NYT Business", "tier": 1},
    {"url": "https://fortune.com/feed/", "name": "Fortune", "tier": 1},

    # === Tier 2: Financial news (usually accessible) ===
    {"url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "name": "CNBC Top News", "tier": 2},
    {"url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069", "name": "CNBC Tech", "tier": 2},
    {"url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147", "name": "CNBC Finance", "tier": 2},
    {"url": "https://feeds.marketwatch.com/marketwatch/topstories", "name": "MarketWatch", "tier": 2},
    {"url": "https://feeds.marketwatch.com/marketwatch/marketpulse", "name": "MarketWatch Pulse", "tier": 2},
    {"url": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best", "name": "Reuters Business", "tier": 2},

    # === Tier 3: Tech / AI ===
    {"url": "https://techcrunch.com/feed/", "name": "TechCrunch", "tier": 3},
    {"url": "https://www.theverge.com/rss/index.xml", "name": "The Verge", "tier": 3},
    {"url": "https://www.wired.com/feed/rss", "name": "Wired", "tier": 3},
    {"url": "https://arstechnica.com/feed/", "name": "Ars Technica", "tier": 3},
    {"url": "https://venturebeat.com/feed/", "name": "VentureBeat", "tier": 3},
    {"url": "https://blogs.nvidia.com/feed/", "name": "NVIDIA Blog", "tier": 3},

    # === Tier 4: Aggregators / Misc ===
    {"url": "https://www.investopedia.com/feedbuilder/feed/getfeed", "name": "Investopedia", "tier": 4},
    {"url": "https://feeds.feedburner.com/TheAihow", "name": "AI How", "tier": 4},
]

# ---------------------------------------------------------------------------
# HTTP Session with retry
# ---------------------------------------------------------------------------

def create_session() -> requests.Session:
    """Create a requests session with retry logic and optional proxy."""
    session = requests.Session()

    # Retry strategy
    retries = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Proxy
    proxies = {}
    if HTTPS_PROXY:
        proxies["https"] = HTTPS_PROXY
    if HTTP_PROXY:
        proxies["http"] = HTTP_PROXY
    if proxies:
        session.proxies.update(proxies)

    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/130.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_time_window() -> tuple[datetime, datetime]:
    """Return (start, end) time window in Beijing time.

    "前一天12点到第二天6点" = previous day noon → today 6 AM (Beijing time).
    """
    now_bj = datetime.now(BEIJING_TZ)
    today_start = now_bj.replace(hour=0, minute=0, second=0, microsecond=0)

    window_end = today_start.replace(hour=WINDOW_END_HOUR)
    window_start = (today_start - timedelta(days=1)).replace(hour=WINDOW_START_HOUR)

    return window_start, window_end


def parse_pub_date(raw: Optional[str]) -> Optional[datetime]:
    """Try multiple formats to parse a published date string."""
    if not raw:
        return None
    raw = raw.strip()
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S",
        "%a, %d %b %Y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            continue
    return None


def article_in_window(published_str: Optional[str],
                      start: datetime,
                      end: datetime) -> bool:
    """Check if article's published time falls within [start, end]."""
    dt = parse_pub_date(published_str)
    if dt is None:
        return True  # include articles we can't parse
    dt_bj = dt.astimezone(BEIJING_TZ)
    return start <= dt_bj <= end


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


# ---------------------------------------------------------------------------
# News Fetcher
# ---------------------------------------------------------------------------

def fetch_rss_articles(start: datetime,
                       end: datetime,
                       session: requests.Session) -> list[dict]:
    """Fetch articles from all RSS feeds within the time window."""
    articles: list[dict] = []
    seen_titles: set[str] = set()
    seen_urls: set[str] = set()

    for feed in RSS_FEEDS:
        name, url, tier = feed["name"], feed["url"], feed["tier"]
        timeout = 20 if tier <= 2 else 15

        try:
            LOG.info(f"  [{tier}] {name}")
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()

            parsed = feedparser.parse(resp.content)
            if not parsed.entries:
                LOG.info(f"    → 0 entries (feed may be stale)")
                continue

            count = 0
            for entry in parsed.entries:
                pub_str = entry.get("published") or entry.get("updated", "")
                if not article_in_window(pub_str, start, end):
                    continue

                title = (entry.get("title") or "").strip()
                link = (entry.get("link") or "").strip()

                # Deduplicate by title AND URL
                if not title:
                    continue
                norm_title = title.lower()
                if norm_title in seen_titles:
                    continue
                if link and link in seen_urls:
                    continue

                seen_titles.add(norm_title)
                if link:
                    seen_urls.add(link)

                summary_raw = entry.get("summary") or entry.get("description") or ""
                articles.append({
                    "title": title,
                    "link": link,
                    "summary": strip_html(summary_raw)[:400],
                    "published": pub_str,
                    "source": name,
                    "source_tier": tier,
                })
                count += 1

            LOG.info(f"    → {count} articles")

        except requests.ConnectionError:
            LOG.info(f"    → unreachable (blocked/timeout)")
        except requests.Timeout:
            LOG.info(f"    → timeout")
        except requests.HTTPError as e:
            LOG.info(f"    → HTTP {e.response.status_code}")
        except Exception as e:
            LOG.info(f"    → error: {e}")

    # Sort by tier (lower = preferred) then by date
    articles.sort(key=lambda a: (a["source_tier"], a.get("published", "")))

    # Limit to MAX_ARTICLES_FOR_LLM
    if len(articles) > MAX_ARTICLES_FOR_LLM:
        articles = articles[:MAX_ARTICLES_FOR_LLM]

    return articles


# ---------------------------------------------------------------------------
# AI Summarizer
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一位专业的财经+科技新闻编辑。你的任务是对提供的新闻列表进行筛选和总结，生成一份中文美国新闻早报。

要求：
1. **📈 美股市场**：总结美国股市相关新闻（大盘趋势、重要公司财报/股价、经济数据、美联储动态等）
2. **🤖 AI与科技**：重点总结AI相关内容，特别是英伟达(NVIDIA)及其供应商（台积电、SK海力士、迈威尔等）相关新闻、AI开发者大会、重要AI产品/模型发布、AI监管政策
3. **📰 其他美国要闻**：地缘政治、宏观经济、科技行业等不在以上两类的美国重要新闻

格式要求：
- 每个分类下用 📌 开头
- 每条新闻 1-3 句话，简洁有力
- 尽量提及具体数据（股价变动百分比、融资金额等）
- 合并同一主题的多篇报道，标注主要来源
- 每条末尾标注 [来源名称]
- 分类内如果没有值得收录的新闻，写"暂无相关要闻"
- 只收录北京时间内发生的新闻

时间范围：北京时间 {start_time} → {end_time}
共收到 {total} 条待筛选新闻。"""


def summarize_news(articles: list[dict],
                   start: datetime,
                   end: datetime) -> str:
    """Use LLM to summarize articles into a Chinese briefing."""
    if not articles:
        return ("## ⚠️ 今日无新闻\n\n"
                f"北京时间 {start.strftime('%m/%d %H:%M')} — "
                f"{end.strftime('%m/%d %H:%M')} 期间未获取到新闻。")

    # Build prompt
    parts = []
    for i, a in enumerate(articles, 1):
        body = f"[{i}] 【{a['source']}】{a['title']}"
        if a.get("summary"):
            body += f"\n    {a['summary'][:300]}"
        if a.get("link"):
            body += f"\n    🔗 {a['link']}"
        parts.append(body)

    article_text = "\n\n".join(parts)
    user_prompt = (
        f"以下是北京时间 {start.strftime('%Y-%m-%d %H:%M')} 至 "
        f"{end.strftime('%Y-%m-%d %H:%M')} 期间收集到的美国新闻"
        f"（共 {len(articles)} 条，已按来源优先级排序）：\n\n"
        f"{article_text}\n\n"
        f"请按以下格式输出中文简报：\n\n"
        f"📈 **美股市场**\n...\n\n"
        f"🤖 **AI与科技**\n...\n\n"
        f"📰 **其他美国要闻**\n...\n"
    )

    try:
        client = Anthropic(
            api_key=ANTHROPIC_API_KEY,
            base_url=ANTHROPIC_BASE_URL,
        )
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=8192,
            system=SYSTEM_PROMPT.format(
                start_time=start.strftime("%Y-%m-%d %H:%M"),
                end_time=end.strftime("%Y-%m-%d %H:%M"),
                total=len(articles),
            ),
            messages=[{"role": "user", "content": user_prompt}],
        )
        # DeepSeek returns ThinkingBlock + TextBlock
        for block in resp.content:
            if hasattr(block, "text"):
                return block.text
        return str(resp.content[0])

    except Exception as e:
        LOG.error(f"LLM error: {e}")
        return _fallback_summary(articles, start, end)


def _fallback_summary(articles: list[dict],
                      start: datetime,
                      end: datetime) -> str:
    """Simple listing without AI."""
    lines = [
        "## 📋 美国新闻简报（无 AI 总结）",
        f"_时间: {start.strftime('%m/%d %H:%M')} — {end.strftime('%m/%d %H:%M')} (北京时间)_",
        f"_共 {len(articles)} 篇新闻_\n",
    ]
    grouped: dict[str, list] = defaultdict(list)
    for a in articles:
        grouped[a["source"]].append(a)

    for source in sorted(grouped):
        items = grouped[source]
        lines.append(f"### {source} ({len(items)}篇)")
        for a in items[:10]:
            lines.append(f"- [{a['title']}]({a['link']})")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feishu Pusher
# ---------------------------------------------------------------------------

def build_feishu_card(markdown: str) -> dict:
    """Build a Feishu interactive card payload."""
    now_str = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M")

    # Truncate if too long
    if len(markdown) > 28000:
        markdown = markdown[:28000] + "\n\n..._(内容过长已截断)_"

    return {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": "🗞️ 美国新闻日报"},
                "template": "blue",
            },
            "elements": [
                {"tag": "markdown", "content": markdown},
                {"tag": "hr"},
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": (
                                f"⏰ {now_str} 北京时间 · "
                                f"自动推送 · "
                                f"来源: CNBC / MarketWatch / TechCrunch / Wired 等"
                            ),
                        }
                    ],
                },
            ],
        },
    }


def push_to_feishu(markdown: str) -> bool:
    """Push card to Feishu. Returns True on success."""
    payload = build_feishu_card(markdown)
    try:
        resp = requests.post(FEISHU_WEBHOOK, json=payload, timeout=20)
        result = resp.json()
        ok = result.get("StatusCode") == 0 or result.get("code") == 0
        if ok:
            LOG.info("✅ 飞书推送成功")
        else:
            LOG.error(f"❌ 飞书返回错误: {result}")
        return ok
    except Exception as e:
        LOG.error(f"❌ 飞书推送异常: {e}")
        return False


def test_feishu() -> bool:
    """Send a test message to verify Feishu connectivity."""
    card = build_feishu_card(
        "## 🧪 飞书连通性测试\n\n"
        "如果你看到这条消息，说明 webhook 工作正常 ✅\n\n"
        f"测试时间: {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M:%S')} (北京时间)"
    )
    try:
        resp = requests.post(FEISHU_WEBHOOK, json=card, timeout=15)
        result = resp.json()
        ok = result.get("StatusCode") == 0 or result.get("code") == 0
        print(f"Feishu test: {'SUCCESS ✅' if ok else 'FAILED ❌'}")
        print(f"Response: {result}")
        return ok
    except Exception as e:
        print(f"Feishu test FAILED: {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # CLI flags
    args = sys.argv[1:]
    if "--debug" in args:
        LOG.setLevel(logging.DEBUG)

    if "--test-feishu" in args:
        return 0 if test_feishu() else 1

    # Custom window
    for arg in args:
        if arg.startswith("--window="):
            parts = arg.split("=")[1].split("-")
            global WINDOW_START_HOUR, WINDOW_END_HOUR
            WINDOW_START_HOUR = int(parts[0])
            WINDOW_END_HOUR = int(parts[1])

    LOG.info("=" * 60)
    LOG.info("🚀 美国新闻每日推送系统")

    # Time window
    start, end = get_time_window()
    LOG.info(f"📅 时间窗口 (北京): {start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')}")

    # HTTP session
    session = create_session()

    # Fetch
    LOG.info("📡 获取 RSS 新闻...")
    articles = fetch_rss_articles(start, end, session)
    LOG.info(f"📊 共获取 {len(articles)} 篇新闻")

    # Summarize
    LOG.info("🤖 AI 总结中...")
    summary = summarize_news(articles, start, end)

    # Save locally
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M")
    log_file = log_dir / f"news_{ts}.md"
    log_file.write_text(summary, encoding="utf-8")
    LOG.info(f"📝 简报已保存: {log_file}")

    # Push
    LOG.info("📤 推送到飞书...")
    success = push_to_feishu(summary)

    LOG.info("🏁 完成!")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
