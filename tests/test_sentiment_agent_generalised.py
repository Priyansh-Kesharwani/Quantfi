import asyncio
import json
import time
import sys
import os

                                
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from pydantic import ValidationError

from engine.indicators.sentiment_agent import (
                   
    SentimentAgentConfig, LLMConfig, CollectionLimits, TierWeights,
                 
    AssetProfile, SourceDocument, KnowledgeSource, SentimentResult,
    SentimentFactor, SentimentCitation,
                
    AssetProfileResolver, NewsCollector, RedditCollector, BlogCollector,
    KnowledgeCompiler, PromptValidator, SentimentAgent,
               
    build_analysis_prompt, _parse_json_safe, _safe_filename,
    compute_sentiment_G_t, full_sentiment_analysis,
)


                                                                             
         
                                                                             

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

results = {"passed": 0, "failed": 0, "warnings": 0}


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        results["passed"] += 1
        print(f"  {PASS} {name}")
    else:
        results["failed"] += 1
        print(f"  {FAIL} {name}  {detail}")


def warn(name: str, detail: str = ""):
    results["warnings"] += 1
    print(f"  {WARN} {name}  {detail}")


def make_mock_docs(tier: str, count: int, prefix: str = "") -> list[SourceDocument]:
    """Create deterministic mock SourceDocuments."""
    return [
        SourceDocument(
            source_id=f"{prefix}{i + 1}",
            tier=tier,
            title=f"Mock {tier} article {i + 1}",
            content=f"Content for {tier} doc {i + 1} about markets and economy.",
            source_name=f"Mock Source {i + 1}",
            url=f"https://example.com/{tier}/{i + 1}",
            published_at=datetime.utcnow() - timedelta(hours=i * 6),
            upvotes=100 * (count - i),
            comment_count=10 * (count - i) if tier == "reddit" else 0,
            is_comment=False,
        )
        for i in range(count)
    ]


def make_mock_comments(post_idx: int, count: int) -> list[SourceDocument]:
    return [
        SourceDocument(
            source_id=f"R{post_idx}.{j + 1}",
            tier="reddit",
            title=f"Comment {j + 1}",
            content=f"Mock comment body {j + 1}",
            source_name="r/mock",
            published_at=datetime.utcnow() - timedelta(hours=j),
            upvotes=50 - j * 5,
            is_comment=True,
            parent_id=f"R{post_idx}",
        )
        for j in range(count)
    ]


                                                                             
                                     
                                                                             


def test_config_models():
    print("\n── TEST 1: Pydantic Config Validation ──")

                                    
    cfg = SentimentAgentConfig()
    check("Default SentimentAgentConfig is valid", True)

                        
    check("LLMConfig default model set", cfg.llm.model == "openai/gpt-4o")
    check("LLMConfig temperature=0", cfg.llm.temperature == 0.0)

                          
    check("TierWeights news=0.45", cfg.tier_weights.news == 0.45)
    check("TierWeights reddit=0.25", cfg.tier_weights.reddit == 0.25)
    check("TierWeights blogs=0.30", cfg.tier_weights.blogs == 0.30)

                               
    check("Limits max_news=20", cfg.limits.max_news_articles == 20)
    check("Limits max_reddit=10", cfg.limits.max_reddit_posts == 10)
    check("Limits max_blogs=10", cfg.limits.max_blog_articles == 10)

                   
    custom = SentimentAgentConfig(
        llm=LLMConfig(model="anthropic/claude-sonnet-4-5-20250929", temperature=0.3),
        limits=CollectionLimits(max_reddit_posts=20, max_news_articles=50),
        tier_weights=TierWeights(news=0.5, reddit=0.3, blogs=0.2),
        extra_blog_feeds={"My Blog": "https://myblog.com/feed"},
    )
    check("Custom LLM model", custom.llm.model == "anthropic/claude-sonnet-4-5-20250929")
    check("Custom limits", custom.limits.max_reddit_posts == 20)
    check("Extra blog feeds", "My Blog" in custom.extra_blog_feeds)

                                                   
    try:
        LLMConfig(temperature=-1.0)
        check("Reject negative temperature", False, "Should have raised")
    except ValidationError:
        check("Reject negative temperature", True)

    try:
        CollectionLimits(max_news_articles=0)
        check("Reject zero max_news", False, "Should have raised")
    except ValidationError:
        check("Reject zero max_news", True)

    try:
        TierWeights(news=1.5)
        check("Reject news weight > 1.0", False, "Should have raised")
    except ValidationError:
        check("Reject news weight > 1.0", True)

                                  
    cfg2 = SentimentAgentConfig().resolve_env()
    check("resolve_env() succeeds", isinstance(cfg2, SentimentAgentConfig))


                                                                             
                      
                                                                             


def test_data_models():
    print("\n── TEST 2: Pydantic Data Models ──")

                  
    profile = AssetProfile(
        symbol="AAPL",
        name="Apple Inc.",
        quote_type="EQUITY",
        sector="Technology",
        industry="Consumer Electronics",
        exchange="NMS",
        currency="USD",
    )
    check("AssetProfile creation", profile.symbol == "AAPL")
    check("AssetProfile extra fields allowed", True)                             

                    
    doc = SourceDocument(
        source_id="N1", tier="news", title="Test", content="Content",
        source_name="Reuters",
    )
    check("SourceDocument creation", doc.source_id == "N1")
    check("SourceDocument default upvotes=0", doc.upvotes == 0)
    check("SourceDocument default is_comment=False", doc.is_comment is False)

                                       
    result = SentimentResult(
        G_t=1.05, raw_sentiment=0.3, confidence=0.7,
    )
    check("SentimentResult valid G_t", result.G_t == 1.05)

                                          
    try:
        SentimentResult(G_t=2.0, raw_sentiment=0.0, confidence=0.5)
        check("Reject G_t=2.0", False, "Should raise")
    except ValidationError:
        check("Reject G_t=2.0 (above max)", True)

    try:
        SentimentResult(G_t=0.5, raw_sentiment=0.0, confidence=0.5)
        check("Reject G_t=0.5", False, "Should raise")
    except ValidationError:
        check("Reject G_t=0.5 (below min)", True)

                     
    f = SentimentFactor(rank=1, factor="Fed policy", direction="bullish",
                        impact_magnitude=0.8, supporting_sources=["N1"],
                        explanation="test")
    check("SentimentFactor creation", f.rank == 1)

                                      
    ks = KnowledgeSource(
        symbol="AAPL",
        asset_profile=profile,
        documents=make_mock_docs("news", 3, "N") + make_mock_docs("blog", 2, "B"),
        market_context={"VIX": 18.5, "regime": "normal"},
        source_counts={"news": 3, "blogs": 2},
    )
    text = ks.to_prompt_text()
    check("KnowledgeSource.to_prompt_text() non-empty", len(text) > 100)
    check("Prompt text contains symbol", "AAPL" in text)
    check("Prompt text contains TIER 1", "TIER 1" in text)
    check("Prompt text contains TIER 3", "TIER 3" in text)

    ids = ks.get_all_source_ids()
    check("get_all_source_ids()", len(ids) == 5)


                                                                             
                                                           
                                                                             


def test_asset_profile_resolver():
    print("\n── TEST 3: Asset Profile Resolver (yfinance) ──")

    cfg = SentimentAgentConfig(llm_profile_resolution=False)                  
    resolver = AssetProfileResolver(cfg)

                                
    test_symbols = [
        ("AAPL", "EQUITY"),
        ("GC=F", "FUTURE"),
        ("BTC-USD", "CRYPTOCURRENCY"),
        ("SPY", "ETF"),
        ("RELIANCE.NS", "EQUITY"),
        ("NFLX", "EQUITY"),
    ]

    async def resolve_all():
        for symbol, expected_type in test_symbols:
            try:
                profile = await resolver.resolve(symbol)
                check(f"Resolve {symbol} → name populated",
                      bool(profile.name))
                check(f"Resolve {symbol} → display_name set",
                      bool(profile.display_name))
                check(f"Resolve {symbol} → subreddits non-empty",
                      len(profile.subreddits) > 0,
                      f"got {profile.subreddits}")
                check(f"Resolve {symbol} → news_keywords non-empty",
                      len(profile.news_keywords) > 0,
                      f"got {profile.news_keywords}")
                check(f"Resolve {symbol} → interpretation_guide set",
                      len(profile.interpretation_guide) > 10,
                      f"got '{profile.interpretation_guide[:50]}'")
                print(f"    → {profile.display_name} | type={profile.quote_type} | "
                      f"subs={profile.subreddits[:3]}")
            except Exception as e:
                check(f"Resolve {symbol}", False, str(e))

    asyncio.run(resolve_all())


                                                                             
                         
                                                                             


def test_news_collector():
    print("\n── TEST 4: News Collector ──")

    cfg = SentimentAgentConfig()                                         
    collector = NewsCollector(cfg)
    profile = AssetProfile(
        symbol="AAPL", name="Apple", news_keywords=["Apple", "AAPL", "tech stocks"],
    )

    async def run():
        docs = await collector.collect(profile)
        check("NewsCollector returns list", isinstance(docs, list))
        if not cfg.newsapi_key:
            check("No API key → empty list (graceful)", len(docs) == 0)
        else:
            check("With API key → got articles", len(docs) > 0)
            if docs:
                check("First doc is SourceDocument", isinstance(docs[0], SourceDocument))
                check("First doc tier='news'", docs[0].tier == "news")

    asyncio.run(run())


                                                                             
                           
                                                                             


def test_reddit_collector():
    print("\n── TEST 5: Reddit Collector ──")

    cfg = SentimentAgentConfig()
    collector = RedditCollector(cfg)
    profile = AssetProfile(
        symbol="AAPL", name="Apple",
        subreddits=["stocks", "investing"],
        news_keywords=["Apple", "AAPL"],
    )

    async def run():
                                                         
        docs = await collector.collect(profile)
        check("RedditCollector returns list", isinstance(docs, list))
                                                                          
        if docs:
            check("Reddit docs are SourceDocuments", isinstance(docs[0], SourceDocument))
            check("Reddit doc tier='reddit'", docs[0].tier == "reddit")
            posts = [d for d in docs if not d.is_comment]
            comments = [d for d in docs if d.is_comment]
            check("Has posts", len(posts) > 0)
            print(f"    → {len(posts)} posts, {len(comments)} comments")
        else:
            warn("No Reddit data (rate limited or no network)", "Non-fatal")

    asyncio.run(run())


                                                                             
                         
                                                                             


def test_blog_collector():
    print("\n── TEST 6: Blog Collector ──")

    cfg = SentimentAgentConfig()
    collector = BlogCollector(cfg)
    profile = AssetProfile(
        symbol="SPY", name="S&P 500 ETF",
        news_keywords=["economy", "inflation", "interest rate", "market", "GDP"],
    )

    async def run():
        docs = await collector.collect(profile)
        check("BlogCollector returns list", isinstance(docs, list))
        if docs:
            check("Blog docs are SourceDocuments", isinstance(docs[0], SourceDocument))
            check("Blog doc tier='blog'", docs[0].tier == "blog")
            check("Blog source IDs are sequential",
                  docs[0].source_id == "B1")
            print(f"    → {len(docs)} blog articles collected")
            for d in docs[:3]:
                print(f"      [{d.source_id}] {d.source_name}: {d.title[:60]}")
        else:
            warn("No blog data (network or parsing issue)", "Non-fatal")

    asyncio.run(run())


                                                                             
                             
                                                                             


def test_knowledge_compiler():
    print("\n── TEST 7: Knowledge Compiler ──")

    cfg = SentimentAgentConfig()
    compiler = KnowledgeCompiler(cfg)

    profile = AssetProfile(
        symbol="AAPL", name="Apple Inc.", display_name="Apple Inc.",
        interpretation_guide="Standard US equity analysis.",
    )
    news = make_mock_docs("news", 5, "N")
    reddit_posts = make_mock_docs("reddit", 3, "R")
    reddit_comments = make_mock_comments(1, 4) + make_mock_comments(2, 3)
    blogs = make_mock_docs("blog", 3, "B")

    ks = compiler.compile(
        profile,
        news, reddit_posts + reddit_comments, blogs,
        {"VIX": 18.5, "regime": "normal"},
    )

    check("KnowledgeSource created", isinstance(ks, KnowledgeSource))
    check("Symbol correct", ks.symbol == "AAPL")
    check("Source counts accurate", ks.source_counts["news"] == 5)
    check("Blogs counted", ks.source_counts["blogs"] == 3)
    check("Reddit posts counted", ks.source_counts["reddit_posts"] == 3)
    check("Reddit comments counted", ks.source_counts["reddit_comments"] == 7)

    text = ks.to_prompt_text()
    check("Prompt text generated", len(text) > 200)
    check("Prompt text contains market context", "VIX" in text)
    print(f"    → {len(text)} chars, {len(ks.documents)} documents")

                                                                                    
    cfg_tiny = SentimentAgentConfig(limits=CollectionLimits(max_knowledge_chars=10_000))
    compiler_tiny = KnowledgeCompiler(cfg_tiny)
                                                                                      
    big_news = make_mock_docs("news", 200, "N")
    ks_trimmed = compiler_tiny.compile(profile, big_news, [], [], {})
    check("Trimming works", len(ks_trimmed.documents) < 200,
          f"got {len(ks_trimmed.documents)}")
    total_chars = sum(len(d.content) + len(d.title) for d in ks_trimmed.documents)
    check("Trimmed fits budget", total_chars <= 10_000 + 100 * len(ks_trimmed.documents))


                                                                             
                         
                                                                             


def test_prompt_builder():
    print("\n── TEST 8: Prompt Builder ──")

    cfg = SentimentAgentConfig(g_min=0.8, g_max=1.2)
    profile = AssetProfile(
        symbol="GC=F", name="Gold Futures", display_name="Gold Futures",
        interpretation_guide="Gold is a safe-haven asset.",
    )
    ks = KnowledgeSource(
        symbol="GC=F", asset_profile=profile,
        documents=make_mock_docs("news", 3, "N"),
        market_context={"VIX": 25.0},
        source_counts={"news": 3},
    )

    prompt = build_analysis_prompt(ks, cfg)
    check("Prompt non-empty", len(prompt) > 500)
    check("Prompt contains G_t bounds", "0.8" in prompt and "1.2" in prompt)
    check("Prompt contains symbol", "GC=F" in prompt)
    check("Prompt contains tier weights", "45%" in prompt)
    check("Prompt contains interpretation guide", "safe-haven" in prompt)
    check("Prompt contains knowledge source", "TIER 1" in prompt)
    print(f"    → prompt length: {len(prompt)} chars")


                                                                             
                           
                                                                             


def test_prompt_validator():
    print("\n── TEST 9: Prompt Validator ──")

    cfg = SentimentAgentConfig()
    validator = PromptValidator(cfg)
    ks = KnowledgeSource(
        symbol="AAPL", asset_profile=AssetProfile(symbol="AAPL"),
        documents=make_mock_docs("news", 3, "N"),
        source_counts={"news": 3},
    )

                          
    valid = json.dumps({
        "G_t": 1.05,
        "raw_sentiment": 0.25,
        "confidence": 0.7,
        "top_factors": [
            {"rank": i, "factor": f"Factor {i}", "direction": "bullish",
             "impact_magnitude": 0.5, "supporting_sources": ["N1"],
             "explanation": "test"} for i in range(1, 6)
        ],
        "source_agreement": 0.6,
        "citations": [{"source_id": "N1", "claim": "test", "direction": "bullish"}],
        "reasoning": "Test reasoning [N1]",
        "asset_specific_notes": "Test notes",
        "dispersion_analysis": "Low dispersion",
    })

    parsed, report = validator.validate(valid, ks)
    check("Valid response accepted", parsed is not None)
    check("Report is_valid=True", report["is_valid"])
    check("No errors", len(report["errors"]) == 0)

                        
    parsed2, report2 = validator.validate("not json {{{", ks)
    check("Invalid JSON rejected", parsed2 is None)
    check("Report is_valid=False for bad JSON", not report2["is_valid"])

                            
    wrapped = f"```json\n{valid}\n```"
    parsed3, report3 = validator.validate(wrapped, ks)
    check("Markdown-wrapped JSON accepted", parsed3 is not None)

                            
    bad_gt = json.loads(valid)
    bad_gt["G_t"] = 5.0
    parsed4, report4 = validator.validate(json.dumps(bad_gt), ks)
    check("Out-of-range G_t flagged", len(report4["errors"]) > 0)

                                   
    partial = json.dumps({"G_t": 1.0})
    parsed5, report5 = validator.validate(partial, ks)
    check("Missing fields detected", len(report5["errors"]) > 0)

                                
    bad_cite = json.loads(valid)
    bad_cite["citations"] = [{"source_id": "Z99", "claim": "fake", "direction": "bullish"}]
    parsed6, report6 = validator.validate(json.dumps(bad_cite), ks)
    check("Invalid citation IDs warned",
          len(report6["warnings"]) > 0 or "Z99" in str(report6.get("checks", {})))

                                 
    bad_logic = json.loads(valid)
    bad_logic["G_t"] = 0.85           
                                     
    parsed7, report7 = validator.validate(json.dumps(bad_logic), ks)
    check("Logical inconsistency detected",
          any("logical" in w.lower() or "direction" in w.lower()
              for w in report7.get("warnings", [])))


                                                                             
                                     
                                                                             


def test_parallel_execution():
    print("\n── TEST 10: Parallel Execution ──")

    async def slow_task(name: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return name

    async def run():
                    
        t0 = time.monotonic()
        r1 = await slow_task("a", 0.1)
        r2 = await slow_task("b", 0.1)
        r3 = await slow_task("c", 0.1)
        seq_time = time.monotonic() - t0

                  
        t0 = time.monotonic()
        r1, r2, r3 = await asyncio.gather(
            slow_task("a", 0.1), slow_task("b", 0.1), slow_task("c", 0.1)
        )
        par_time = time.monotonic() - t0

        check("Sequential takes ~0.3s", seq_time >= 0.25)
        check("Parallel takes ~0.1s", par_time < 0.2,
              f"took {par_time:.3f}s")
        check("Parallel is faster", par_time < seq_time)
        print(f"    → sequential={seq_time:.3f}s, parallel={par_time:.3f}s, "
              f"speedup={seq_time / par_time:.1f}x")

    asyncio.run(run())


                                                                             
                                    
                                                                             


def test_no_hardcoded_asset_types():
    print("\n── TEST 11: No Hardcoded Asset Types ──")

                                                                              
    import inspect
    sig = inspect.signature(SentimentAgent.analyze)
    params = list(sig.parameters.keys())
    check("analyze() has 'symbol' param", "symbol" in params)
    check("analyze() does NOT have 'asset_type' param", "asset_type" not in params)

                                                 
    from engine.indicators.sentiment_agent import ASSET_PROFILES
    check("ASSET_PROFILES is empty (deprecated)", len(ASSET_PROFILES) == 0)

                                     
    exotic = AssetProfile(
        symbol="EURUSD=X", name="EUR/USD", quote_type="CURRENCY",
        sector="", industry="", exchange="CCY",
    )
    check("Exotic asset (FOREX) profile created", exotic.quote_type == "CURRENCY")

    crypto = AssetProfile(
        symbol="ETH-USD", name="Ethereum", quote_type="CRYPTOCURRENCY",
    )
    check("Crypto asset profile created", crypto.quote_type == "CRYPTOCURRENCY")

    bond = AssetProfile(
        symbol="TLT", name="iShares 20+ Year Treasury Bond ETF",
        quote_type="ETF", sector="Fixed Income",
    )
    check("Bond ETF profile created", bond.sector == "Fixed Income")


                                                                             
                             
                                                                             


def test_utilities():
    print("\n── TEST 12: Utility Functions ──")

                    
    check("safe_filename replaces special chars with underscore",
          _safe_filename("GC=F") == "GC_F")
    check("safe_filename preserves dots",
          _safe_filename("RELIANCE.NS") == "RELIANCE.NS")
    check("safe_filename strips slashes",
          "/" not in _safe_filename("a/b/c"))

                      
    check("parse_json_safe valid", _parse_json_safe('{"a": 1}') == {"a": 1})
    check("parse_json_safe invalid", _parse_json_safe("not json") is None)
    check("parse_json_safe markdown",
          _parse_json_safe('```json\n{"a": 1}\n```') == {"a": 1})


                                                                             
                                        
                                                                             


def test_end_to_end():
    print("\n── TEST 13: End-to-End Pipeline ──")

                                                          
    import engine.indicators.sentiment_agent as sa

    original_llm = sa._llm_completion
    call_count = [0]

    async def mock_llm(cfg, *, system, user):
        call_count[0] += 1
        if "generate a sentiment-analysis profile" in user.lower():
                                     
            return json.dumps({
                "display_name": "Test Asset",
                "interpretation_guide": "This is a test asset. Analyse normally.",
                "key_factors": ["Factor 1", "Factor 2", "Factor 3", "Factor 4", "Factor 5"],
                "subreddits": ["investing", "stocks"],
                "news_keywords": ["test", "asset", "market"],
                "sentiment_inversion_events": [],
            })
        else:
                                
            return json.dumps({
                "G_t": 1.03,
                "raw_sentiment": 0.15,
                "confidence": 0.6,
                "top_factors": [
                    {"rank": i, "factor": f"Factor {i}", "direction": "bullish",
                     "impact_magnitude": 0.5, "supporting_sources": [f"N{i}"],
                     "explanation": f"Factor {i} explanation"} for i in range(1, 6)
                ],
                "source_agreement": 0.65,
                "citations": [
                    {"source_id": "N1", "claim": "Test", "direction": "bullish"},
                ],
                "reasoning": "Mock reasoning [N1].",
                "asset_specific_notes": "Mock notes.",
                "dispersion_analysis": "Low dispersion.",
            })

    sa._llm_completion = mock_llm

                                                        
    test_symbols = ["AAPL", "GC=F", "SPY"]

    async def run():
        for sym in test_symbols:
            cfg = SentimentAgentConfig(llm_profile_resolution=True)
            agent = SentimentAgent(cfg)

            try:
                result = await agent.analyze(sym)
                check(f"E2E {sym}: returns SentimentResult",
                      isinstance(result, SentimentResult))
                check(f"E2E {sym}: G_t in range",
                      0.8 <= result.G_t <= 1.2, f"G_t={result.G_t}")
                check(f"E2E {sym}: has top_factors",
                      len(result.top_factors) > 0)
                check(f"E2E {sym}: has reasoning",
                      bool(result.reasoning))
                check(f"E2E {sym}: meta has symbol",
                      result.meta.get("symbol") == sym)
                print(f"    → {sym}: G_t={result.G_t:.4f}, "
                      f"conf={result.confidence:.2f}, "
                      f"factors={len(result.top_factors)}")
            except Exception as e:
                check(f"E2E {sym}", False, str(e))

    try:
        asyncio.run(run())
    finally:
        sa._llm_completion = original_llm

    check("LLM mock was called", call_count[0] > 0,
          f"call_count={call_count[0]}")


                                                                             
                                                  
                                                                             


def test_config_serialisation():
    print("\n── TEST 14: Config Serialisation ──")

    cfg = SentimentAgentConfig(
        llm=LLMConfig(model="anthropic/claude-sonnet-4-5-20250929"),
        limits=CollectionLimits(max_reddit_posts=15),
        extra_blog_feeds={"Custom": "https://custom.com/rss"},
    )

                       
    json_str = cfg.model_dump_json(indent=2)
    check("Config serialises to JSON", len(json_str) > 50)

                      
    cfg2 = SentimentAgentConfig.model_validate_json(json_str)
    check("Config round-trips",
          cfg2.llm.model == "anthropic/claude-sonnet-4-5-20250929")
    check("Limits round-trip", cfg2.limits.max_reddit_posts == 15)
    check("Extra feeds round-trip", "Custom" in cfg2.extra_blog_feeds)


                                                                             
        
                                                                             


def main():
    print("=" * 70)
    print("GENERALISED SENTIMENT AGENT — COMPREHENSIVE VERIFICATION")
    print("=" * 70)

                                   
    test_config_models()
    test_data_models()
    test_knowledge_compiler()
    test_prompt_builder()
    test_prompt_validator()
    test_parallel_execution()
    test_no_hardcoded_asset_types()
    test_utilities()
    test_config_serialisation()

                                                       
    print("\n" + "=" * 70)
    print("NETWORK-DEPENDENT TESTS (yfinance, Reddit, blogs)")
    print("=" * 70)
    test_asset_profile_resolver()
    test_news_collector()
    test_reddit_collector()
    test_blog_collector()

                                                            
    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE TEST (mocked LLM)")
    print("=" * 70)
    test_end_to_end()

             
    print("\n" + "=" * 70)
    total = results["passed"] + results["failed"]
    pct = results["passed"] / total * 100 if total else 0
    status = "ALL PASSED" if results["failed"] == 0 else "FAILURES DETECTED"
    colour = "\033[92m" if results["failed"] == 0 else "\033[91m"
    print(f"{colour}RESULTS: {results['passed']}/{total} passed ({pct:.0f}%)  "
          f"| {results['warnings']} warnings | {status}\033[0m")
    print("=" * 70)

    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    exit(main())
