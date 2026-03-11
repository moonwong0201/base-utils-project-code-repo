"""基于 asyncio + aiohttp 实现的异步高并发压测"""
import time
import asyncio
import aiohttp
import statistics


async def test_once(session, url, message, session_id, sem):
    """单次请求"""
    async with sem:
        start = time.perf_counter()
        try:
            async with session.post(
                    url,
                    json={"session_id": session_id, "message": message},
                    timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                data = await resp.json()

            latency = (time.perf_counter() - start) * 1000
            from_cache = data.get("from_cache", False)
            route = data.get("route")

            return {
                "latency": latency,
                "from_cache": from_cache,
                "route": route,
                "success": True,
                "message": message
            }

        except Exception as e:
            return {
                "latency": 0,
                "from_cache": False,
                "route": None,
                "success": False,
                "error": str(e),
                "message": message
            }


async def run_scenario(session, url, sem, messages, total, scenario_name):
    """运行单个场景"""
    print(f"\n{'=' * 50}")
    print(f"场景: {scenario_name}")
    print(f"总请求数: {total}")
    print(f"{'=' * 50}")

    tasks = []
    start_time = time.perf_counter()

    for i in range(total):
        msg = messages[i % len(messages)]
        # 每个请求用不同的session，避免跨会话污染
        task = test_once(
            session,
            url,
            msg,
            f"session_{i % 10}",  # 300个不同session，强制每条消息都是"首次"
            sem
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    successes = [r for r in results if r["success"]]
    n = len(successes)

    if not successes:
        print("全部请求失败")
        return None

    latencies = sorted(r["latency"] for r in successes)
    cache_hits = sum(1 for r in successes if r["from_cache"])
    route_hits = sum(1 for r in successes if r["route"] != "Other")

    # 按消息类型统计命中率
    msg_stats = {}
    for r in successes:
        msg = r["message"]
        if msg not in msg_stats:
            msg_stats[msg] = {"total": 0, "hits": 0}
        msg_stats[msg]["total"] += 1
        if r["from_cache"]:
            msg_stats[msg]["hits"] += 1

    print("\n各消息命中情况:")
    for msg, stats in msg_stats.items():
        rate = stats["hits"] / stats["total"] * 100
        print(f"  {msg[:25]:25} | 请求{stats['total']:3}次 | 命中{stats['hits']:3}次 | {rate:5.1f}%")

    return {
        "scenario": scenario_name,
        "total": total,
        "success": n,
        "time": total_time,
        "qps": n / total_time,
        "avg_latency": statistics.mean(latencies),
        "p95": latencies[int(n * 0.95)] if n > 1 else 0,
        "p99": latencies[int(n * 0.99)] if n > 1 else 0,
        "cache_hit_rate": cache_hits / n * 100,
        "route_hit_rate": route_hits / n * 100,
    }


def print_result(r):
    """打印结果"""
    if not r:
        return
    print(f"\n结果:")
    print(f"  总请求: {r['total']}")
    print(f"  成功: {r['success']}")
    print(f"  QPS: {r['qps']:.2f}")
    print(f"  平均延迟: {r['avg_latency']:.1f}ms")
    print(f"  P95延迟: {r['p95']:.1f}ms")
    print(f"  P99延迟: {r['p99']:.1f}ms")
    print(f"  缓存命中率: {r['cache_hit_rate']:.1f}%")
    print(f"  路由命中率: {r['route_hit_rate']:.1f}%")


async def benchmark():
    url = "http://localhost:8000/chat"
    concurrency = 50
    sem = asyncio.Semaphore(concurrency)

    # ========= 先添加路由 =========
    routes = [
        ("Travel-Query", ["还有双鸭山到淮阴的汽车票吗"]),
        ("Music-Play", ["播放钢琴曲命运交响曲"]),
        ("Video-Play", ["给我找一个魔兽世界的比赛视频"]),
        ("Weather-Query", ["海南今天几级风"]),
    ]

    # ========= 定义测试场景 =========
    scenario_100_hit = {
        "name": "100%缓存命中",
        "messages": ["还有双鸭山到淮阴的汽车票吗"] * 6,  # 6个都是预热过的
        "total": 300,
        "warmup": ["还有双鸭山到淮阴的汽车票吗", "播放钢琴曲命运交响曲", "海南今天几级风"],
    }

    scenario_70_hit = {
        "name": "70%缓存命中（40%精准+30%语义+30%不命中）",
        "total": 300,
        "messages": [
            # 40% 精准命中（预热过，每次查询都命中）
            "还有双鸭山到淮阴的汽车票吗",
            "播放钢琴曲命运交响曲",
            "给我找一个魔兽世界的比赛视频",
            "海南今天几级风",

            # 30% 语义相似（Faiss模糊匹配会命中预热的向量）
            "双鸭山到淮阴的汽车票查询一下",
            "命运交响曲钢琴曲播放一下",
            "魔兽世界比赛视频找一下",

            # 30% 不命中（全新意图，和预热完全不同）
            "今天股票行情怎么样",
            "推荐一款性价比高的手机",
            "Python怎么学习比较好",
        ],
        # 只预热前4条
        "warmup": [
            "还有双鸭山到淮阴的汽车票吗",
            "播放钢琴曲命运交响曲",
            "给我找一个魔兽世界的比赛视频",
            "海南今天几级风",
        ],
    }

    # 场景3: 0%缓存命中（全冷启动）
    scenario_0_hit = {
        "name": "0%缓存命中(冷启动)",
        "messages": [f"随机问题{i}" for i in range(6)],
        "total": 30,
        "warmup": [],
    }

    # scenarios = [scenario_100_hit, scenario_70_hit]
    scenarios = [scenario_100_hit]
    all_results = []

    for scenario in scenarios:
        # 清空所有缓存（包括路由）
        print(f"\n清空缓存...")
        async with aiohttp.ClientSession() as session:
            await session.post("http://localhost:8000/cache/clear?clear_router=true")

        print("添加路由...")
        async with aiohttp.ClientSession() as session:
            for target, questions in routes:
                await session.post(
                    "http://localhost:8000/router/add",
                    json={"target": target, "questions": questions}
                )
                print(f"  添加路由: {target}")

        # 预热
        if scenario["warmup"]:
            print(f"预热: {scenario['warmup']}")
            async with aiohttp.ClientSession() as session:
                for msg in scenario["warmup"]:
                    await session.post(
                        url,
                        json={"session_id": "warmup", "message": msg}
                    )

        # 压测
        async with aiohttp.ClientSession() as session:
            result = await run_scenario(
                session,
                url,
                sem,
                scenario["messages"],
                scenario["total"],
                scenario["name"]
            )
            print_result(result)
            if result:
                all_results.append(result)

    # ========= 汇总 =========
    print(f"\n{'=' * 50}")
    print("汇总对比")
    print(f"{'=' * 50}")
    print(f"{'场景':<20} {'QPS':>8} {'延迟(ms)':>10} {'缓存命中':>10} {'路由命中':>10}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['scenario']:<20} {r['qps']:>8.1f} {r['avg_latency']:>10.1f} {r['cache_hit_rate']:>9.1f}% {r['route_hit_rate']:>9.1f}%")


if __name__ == "__main__":
    asyncio.run(benchmark())

