# 压测
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
                "success": True
            }

        except Exception as e:
            return {
                "latency": 0,
                "from_cache": False,
                "route": None,
                "success": False,
                "error": str(e)
            }


async def run_scenario(session, url, sem, messages, total, scenario_name):
    """运行单个场景"""
    print(f"\n{'='*50}")
    print(f"场景: {scenario_name}")
    print(f"{'='*50}")

    tasks = []
    start_time = time.perf_counter()

    for i in range(total):
        msg = messages[i % len(messages)]
        task = test_once(
            session,
            url,
            msg,
            f"session_{i % 10}",
            sem
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()

    # 统计
    total_time = end_time - start_time
    successes = [r for r in results if r["success"]]
    n = len(successes)

    if not successes:
        print("全部请求失败")
        return None

    latencies = sorted(r["latency"] for r in successes)
    cache_hits = sum(1 for r in successes if r["from_cache"])
    route_hits = sum(1 for r in successes if r["route"] is not None)

    return {
        "scenario": scenario_name,
        "total": total,
        "success": n,
        "time": total_time,
        "qps": n / total_time,
        "avg_latency": statistics.mean(latencies),
        "p95": latencies[int(n * 0.95)],
        "p99": latencies[int(n * 0.99)],
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
    print("初始化路由...")
    async with aiohttp.ClientSession() as session:
        routes = [
            ("refund", ["怎么退款", "如何退货", "我想退款"]),
            ("greeting", ["你好", "您好", "hello", "hi"]),
            ("order", ["查询订单", "我的订单", "订单状态"]),
            ("shipping", ["物流查询", "快递到哪了", "发货了吗"]),
        ]
        for target, questions in routes:
            await session.post(
                "http://localhost:8000/router/add",
                json={"target": target, "questions": questions}
            )
            print(f"  添加路由: {target}")

    # ========= 定义测试场景 =========

    # 场景1: 100%缓存命中
    scenario_100_hit = {
        "name": "100%缓存命中",
        "messages": ["怎么退款"] * 6,  # 6个都是预热过的
        "total": 300,
        "warmup": ["怎么退款", "如何退货", "我想退款"],
    }

    # 场景2: 70%缓存命中 混合场景
    scenario_70_hit = {
        "name": "70%缓存命中",
        "messages": [
            # 精准命中
            "怎么退款",  
            "你好",  
            "hello",  
            "查询订单",  

            # 语义相似
            "如何申请退款",
            "您好",
            "我的订单在哪",

            # 完全不相关
            "今天天气如何",
            "推荐餐厅",
            "怎么学习Python",
        ],
        "total": 300,
        "warmup": ["怎么退款", "如何退货", "我想退款", "你好", "查询订单"],
    }

    # 场景3: 0%缓存命中
    scenario_0_hit = {
        "name": "0%缓存命中(冷启动)",
        "messages": [f"随机问题{i}" for i in range(6)],
        "total": 30,
        "warmup": [],
    }

    scenarios = [scenario_100_hit, scenario_70_hit, scenario_0_hit]

    # ========= 运行所有场景 =========
    all_results = []

    for scenario in scenarios:
        # 清空缓存
        print(f"\n清空缓存...")
        async with aiohttp.ClientSession() as session:
            await session.post("http://localhost:8000/cache/clear?clear_router=false")

        # 预热
        if scenario["warmup"]:
            print(f"预热: {scenario['warmup']}")
            async with aiohttp.ClientSession() as session:
                for msg in scenario["warmup"]:
                    await session.post(
                        url,
                        json={"session_id": "warmup", "message": msg}
                    )

        # 运行压测
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
    print(f"\n{'='*50}")
    print("汇总对比")
    print(f"{'='*50}")
    print(f"{'场景':<20} {'QPS':>8} {'延迟(ms)':>10} {'缓存命中':>10} {'路由命中':>10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['scenario']:<20} {r['qps']:>8.1f} {r['avg_latency']:>10.1f} {r['cache_hit_rate']:>9.1f}% {r['route_hit_rate']:>9.1f}%")


if __name__ == "__main__":
    asyncio.run(benchmark())
