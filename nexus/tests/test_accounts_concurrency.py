"""并发抢占评测：多 ExecutionContext 同时争抢账号池

测试维度：
  A. 多 ctx 同池并发 acquire —— 验证服务端互斥正确
  B. 单 ctx 内多协程并发 acquire —— 验证 _accounts / _heartbeat_tasks 无竞态
  C. ctx 退出自动 release 的并发安全性
  D. 独占/共享 混合并发
  E. 压力测试：100 请求 10 账号
"""

import asyncio
import os
import sys
import time
import uuid

import pytest

from nexus.runtime.accounts import AccountPoolManager, NoAvailableAccountError


TAG = uuid.uuid4().hex[:6]
EXCL_POOL = f'concur_excl_{TAG}'
SHR_POOL = f'concur_shr_{TAG}'
EXCL_COUNT = 20  # 独占账号数量
SHR_COUNT = 5
SHR_CAP = 10  # 每个共享账号 cap
CONCURRENT_GRABBERS = 40  # 并发抢号数
# 远端服务器地址。默认指向生产账号池，所以 pytest 下必须显式开启才会跑。
ENDPOINT = os.environ.get('NEXUS_ACCOUNT_POOL_ENDPOINT', 'http://nexus-pool.production.polaris:8080')
RUN_ENV = 'NEXUS_RUN_ACCOUNT_POOL_CONCURRENCY'

# 这些用例会在真实账号池服务上创建/删除账号并发起 40 并发抢占，默认不在 `pytest tests`
# 里执行。直接 `python tests/test_accounts_concurrency.py` 走 main() 不受此门禁影响。
pytestmark = pytest.mark.skipif(
    os.environ.get(RUN_ENV) != '1',
    reason=f'set {RUN_ENV}=1 to run account pool concurrency tests against a live pool service',
)

PASS = FAIL = 0
created_ids = []


def banner(title: str):
    print(f'\n{"=" * 60}')
    print(f'  {title}')
    print(f'{"=" * 60}')


async def setup(mgr: AccountPoolManager):
    """创建测试账号。"""
    for i in range(EXCL_COUNT):
        aid = await mgr.add_account(EXCL_POOL, f'e{i:03d}', credentials={'pwd': str(i)})
        created_ids.append(aid)
    for i in range(SHR_COUNT):
        aid = await mgr.add_account(SHR_POOL, f's{i:03d}', credentials={'pwd': str(i)}, shared_cap=SHR_CAP)
        created_ids.append(aid)
    print(f'  setup: {EXCL_COUNT} 独占 + {SHR_COUNT} 共享(cap={SHR_CAP}) 已创建')


async def cleanup(mgr: AccountPoolManager):
    await mgr.release_all()
    for aid in sorted(set(created_ids)):
        try:
            await mgr.delete_account(aid)
        except Exception:
            pass
    await mgr.aclose()


@pytest.fixture(scope='module', autouse=True)
def _provision_account_pools():
    """在 pytest 下复刻 main() 的 setup/cleanup。

    每个用例自己 new AccountPoolManager（各自绑定到 function 级事件循环），所以这里用
    同步 fixture + asyncio.run() 建一个一次性的循环来建池/销毁，不会跨循环复用 session。

    注意：账号池服务没有对外暴露建池接口（client 只有 AddAccount/Acquire/GetPoolStats
    等），池必须由服务端预先开好。而 EXCL_POOL/SHR_POOL 带随机 TAG，因此在未预置这些池
    的环境上 AddAccount 会直接返回 `pool "xxx" not found`——此时跳过而不是报错，避免把
    环境缺失伪装成代码缺陷。
    """

    async def _provision():
        mgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=60)
        try:
            await setup(mgr)
        finally:
            await mgr.aclose()

    async def _teardown():
        mgr = AccountPoolManager(endpoint=ENDPOINT)
        try:
            await mgr.release_all()
            await asyncio.sleep(2)  # 等 reaper
            for aid in sorted(set(created_ids)):
                try:
                    await mgr.delete_account(aid)
                except Exception:
                    pass
        finally:
            await mgr.aclose()

    try:
        asyncio.run(_provision())
    except Exception as exc:
        try:
            asyncio.run(_teardown())
        except Exception:  # noqa: BLE001 — teardown best-effort; skip must always fire
            pass
        pytest.skip(f'cannot provision pools {EXCL_POOL}/{SHR_POOL} on {ENDPOINT}: {exc}')

    try:
        yield
    finally:
        asyncio.run(_teardown())


# ============================================================================
# 测试 A：多 ctx 同池并发 acquire —— 验证互斥
# ============================================================================


async def test_a_multi_ctx_concurrent():
    """40 个 ctx 同时抢 20 个独占账号，每个 ctx 只抢 1 个。"""
    global PASS, FAIL
    banner('A: 多 ctx 同池并发 acquire（40 ctx 抢 20 账号）')

    results = {'acquired': 0, 'no_avail': 0, 'error': 0, 'dup_ids': set(), 'all_ids': []}

    async def grab_worker(i: int):
        """每个 worker 创建独立 ctx（独立 AccountPoolManager）。"""
        mgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=30)
        try:
            act = await mgr.acquire(EXCL_POOL, ttl_seconds=30)
            # 记录获取到的 account_id
            results['acquired'] += 1
            results['all_ids'].append(act.account_id)
            # 检查是否重复（同一账号不应被两次 acquire）
            if act.account_id in results['dup_ids']:
                results['error'] += 1
                print(f'  ⚠️  DUPLICATE! worker={i} got account_id={act.account_id} (already acquired)')
            results['dup_ids'].add(act.account_id)
            # 保留 mgr 交给外层统一 release+aclose，避免 aiohttp session 泄漏
            return {'ok': True, 'account_id': act.account_id, 'lease_id': act.lease_id, 'mgr': mgr}
        except NoAvailableAccountError:
            results['no_avail'] += 1
            await mgr.aclose()
            return {'ok': False, 'reason': 'no_avail'}
        except Exception as exc:
            results['error'] += 1
            print(f'  ⚠️  worker={i} error: {exc}')
            await mgr.aclose()
            return {'ok': False, 'reason': str(exc)}

    # 40 个 worker 并发抢
    tasks = [grab_worker(i) for i in range(CONCURRENT_GRABBERS)]
    grabs = await asyncio.gather(*tasks)

    # 验证
    acquired_leases = [g for g in grabs if g['ok']]

    stat_mgr = AccountPoolManager(endpoint=ENDPOINT)
    try:
        stats = {s.pool: s for s in await stat_mgr.get_pool_stats()}
    finally:
        await stat_mgr.aclose()
    s = stats[EXCL_POOL]

    print(f'  成功获取: {results["acquired"]}')
    print(f'  无可用账号: {results["no_avail"]}')
    print(f'  错误: {results["error"]}')
    print(f'  去重后 account_id 数: {len(results["dup_ids"])}')
    print(f'  服务端状态: idle={s.exclusive_idle} acquired={s.exclusive_acquired}')

    # 断言：最多 20 个成功（只有 20 个账号），且无重复
    ok = (
        results['acquired'] <= EXCL_COUNT
        and len(results['dup_ids']) == results['acquired']
        and results['error'] == 0
        and s.exclusive_acquired == results['acquired']
        and s.exclusive_idle == EXCL_COUNT - results['acquired']
    )
    if ok:
        PASS += 1
        print('  ✅ [PASS] A')
    else:
        FAIL += 1
        print('  ❌ [FAIL] A')

    # 统一清理：复用 worker 自己的 mgr 释放对应 lease，然后 aclose
    for g in acquired_leases:
        wmgr = g['mgr']
        try:
            await wmgr.release(g['lease_id'])
        except Exception:
            pass
        finally:
            await wmgr.aclose()
    return acquired_leases


# ============================================================================
# 测试 B：单 ctx 内多协程并发 acquire
# ============================================================================


async def test_b_single_ctx_internal_concurrency():
    """同一 AccountPoolManager 内 30 个协程并发 acquire 20 个账号。"""
    global PASS, FAIL
    banner('B: 单 ctx 内多协程并发 acquire（30 协程抢 20 账号）')

    mgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=30)
    dup_check = set()
    errors = []

    async def inner_grab(i: int):
        try:
            act = await mgr.acquire(EXCL_POOL, ttl_seconds=30)
            if act.account_id in dup_check:
                errors.append(f'DUPLICATE worker={i} id={act.account_id}')
            dup_check.add(act.account_id)
            return act
        except NoAvailableAccountError:
            return None
        except Exception as exc:
            errors.append(f'worker={i}: {exc}')
            return None

    acts = await asyncio.gather(*[inner_grab(i) for i in range(30)])
    acquired = [a for a in acts if a is not None]

    print(f'  成功获取: {len(acquired)}')
    print(f'  无可用账号: {len(acts) - len(acquired)}')
    print(f'  去重后 account_id: {len(dup_check)}')
    if errors:
        for e in errors:
            print(f'  ⚠️  {e}')

    ok = len(acquired) == EXCL_COUNT and len(dup_check) == EXCL_COUNT and not errors
    if ok:
        PASS += 1
        print('  ✅ [PASS] B')
    else:
        FAIL += 1
        print('  ❌ [FAIL] B')

    # 验证 _accounts 和 _heartbeat_tasks 状态一致
    print(f'  _accounts keys: {len(mgr._accounts)}')
    print(f'  _heartbeat_tasks keys: {len(mgr._heartbeat_tasks)}')

    await mgr.release_all()
    print(f'  清理后 _accounts: {len(mgr._accounts)}, _heartbeat_tasks: {len(mgr._heartbeat_tasks)}')

    ok2 = len(mgr._accounts) == 0 and len(mgr._heartbeat_tasks) == 0
    if not ok2:
        print('  ⚠️  _accounts / _heartbeat_tasks 未完全清理!')
    await mgr.aclose()


# ============================================================================
# 测试 C：ctx 退出自动 release + 并发 ctx
# ============================================================================


async def test_c_ctx_cleanup_concurrency():
    """多个 ctx 各自 acquire 后退出，验证 release 并发安全性。"""
    global PASS, FAIL
    banner('C: 多 ctx 退出自动 release 并发（20 ctx × 1 账号）')

    # 先确保全部 idle
    mgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=10)
    stats = {s.pool: s for s in await mgr.get_pool_stats()}
    s = stats[EXCL_POOL]
    if s.exclusive_idle < EXCL_COUNT:
        print(f'  ⚠️  idle 不足 ({s.exclusive_idle}), 可能是上轮残留，等 reaper...')
        await asyncio.sleep(2)
    await mgr.aclose()

    ctx_results = []

    async def ctx_lifecycle(i: int, pool: str):
        mgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=10)
        try:
            act = await mgr.acquire(pool, ttl_seconds=10)
            ctx_results.append({'worker': i, 'id': act.account_id, 'lease': act.lease_id})
            # 模拟短暂使用
            await asyncio.sleep(0.05)
        except NoAvailableAccountError:
            ctx_results.append({'worker': i, 'id': None})
        finally:
            await mgr.release_all()
            await mgr.aclose()

    # 20 个 ctx 并发
    await asyncio.gather(*[ctx_lifecycle(i, EXCL_POOL) for i in range(20)])

    acquired = [r for r in ctx_results if r['id'] is not None]
    no_avail = [r for r in ctx_results if r['id'] is None]
    print(f'  成功获取: {len(acquired)}, 无账号: {len(no_avail)}')

    # 最终全部应该回到 idle
    mgr2 = AccountPoolManager(endpoint=ENDPOINT)
    try:
        stats = {s.pool: s for s in await mgr2.get_pool_stats()}
        s = stats[EXCL_POOL]
    finally:
        await mgr2.aclose()
    print(f'  最终状态: idle={s.exclusive_idle} acquired={s.exclusive_acquired}')

    ok = s.exclusive_idle == EXCL_COUNT and s.exclusive_acquired == 0 and len(no_avail) == 0
    if ok:
        PASS += 1
        print('  ✅ [PASS] C')
    else:
        FAIL += 1
        print('  ❌ [FAIL] C')


# ============================================================================
# 测试 D：独占/共享 混合并发
# ============================================================================


async def test_d_mixed_concurrency():
    """独占和共享账号同时被不同 ctx 争抢。"""
    global PASS, FAIL
    banner(f'D: 独占/共享混合并发（各 {SHR_CAP * 2} 个 grabber）')

    async def grab(i: int, pool: str, tag: str):
        """通用抓取器：无论后续如何，都保证 mgr 得到 release_all + close。

        - 成功时：把 mgr 一并带出，由外层统一 release_all + close；
        - 拿不到账号（NoAvailableAccountError）：本函数内 release_all + close，返回 None；
        - 其他异常：本函数内 release_all + close，返回 {'error': ...}。
        """
        mgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=30)
        try:
            act = await mgr.acquire(pool, ttl_seconds=30)
            return {'pool': tag, 'worker': i, 'id': act.account_id, 'lease': act.lease_id, 'mgr': mgr}
        except NoAvailableAccountError:
            try:
                await mgr.release_all()
            finally:
                await mgr.aclose()
            return None
        except Exception as exc:
            try:
                await mgr.release_all()
            finally:
                await mgr.aclose()
            return {'error': str(exc)}

    total_grabbers = SHR_CAP * SHR_COUNT + 5  # 共享容量 + 超额
    tasks = []
    for i in range(total_grabbers):
        tasks.append(grab(i, EXCL_POOL, 'excl'))  # 抢独占
        tasks.append(grab(i, SHR_POOL, 'shr'))  # 抢共享

    results = await asyncio.gather(*tasks)

    excl_ok = [r for r in results if r and r.get('pool') == 'excl']
    shr_ok = [r for r in results if r and r.get('pool') == 'shr']
    errors = [r for r in results if r and r.get('error')]
    excl_dup = len({r['id'] for r in excl_ok})

    print(f'  独占获取: {len(excl_ok)} (去重: {excl_dup})')
    print(f'  共享获取: {len(shr_ok)}')
    if errors:
        for e in errors[:5]:
            print(f'  ⚠️  grabber error: {e["error"]}')

    # 清理：所有带 mgr 的都必须 release_all + close，即便断言失败也不能漏
    for r in excl_ok + shr_ok:
        try:
            await r['mgr'].release_all()
        except Exception as exc:
            print(f'  ⚠️  release_all failed for lease={r.get("lease")}: {exc}')
        finally:
            try:
                await r['mgr'].aclose()
            except Exception:
                pass

    # 校验服务端状态是否干净（SDK 层已做 exception-safe 兜底，无需 invalidate/recover）
    verify_mgr = AccountPoolManager(endpoint=ENDPOINT)
    try:
        stats = {s.pool: s for s in await verify_mgr.get_pool_stats()}
        excl_stat = stats[EXCL_POOL]
        shr_stat = stats[SHR_POOL]
        print(
            f'  最终状态 excl: idle={excl_stat.exclusive_idle} '
            f'acquired={excl_stat.exclusive_acquired} reserved={excl_stat.exclusive_reserved}'
        )
        print(
            f'  最终状态 shr : capacity={shr_stat.shared_capacity} '
            f'occupied={shr_stat.shared_occupied} invalid={shr_stat.shared_invalid}'
        )
    finally:
        await verify_mgr.aclose()

    clean = (
        excl_stat.exclusive_acquired == 0
        and excl_stat.exclusive_reserved == 0
        and excl_stat.exclusive_idle == EXCL_COUNT
        and shr_stat.shared_occupied == 0
        and shr_stat.shared_invalid == 0
    )

    # 验证独占无重复，共享不超过总容量，且清理后完全干净
    ok = (
        excl_dup == len(excl_ok)  # 独占无重复
        and len(excl_ok) <= EXCL_COUNT  # 不超过总数
        and len(shr_ok) <= SHR_COUNT * SHR_CAP  # 不超过共享容量
        and not errors  # 无 grabber 异常
        and clean  # release_all 后服务端干净
    )
    if ok:
        PASS += 1
        print('  ✅ [PASS] D')
    else:
        FAIL += 1
        if not clean:
            print('  ❌ [FAIL] D — release_all 后服务端仍不干净（存在孤儿 lease）')
        else:
            print('  ❌ [FAIL] D')


# ============================================================================
# 测试 E：压力测试 100 请求 10 账号
# ============================================================================


async def test_e_stress():
    """压力测试：100 个并发请求 vs 10 个账号（独占），测量延迟和正确性。"""
    global PASS, FAIL
    EXCL_STRESS = 10
    TOTAL_REQUESTS = 100
    banner(f'E: 压力测试（{TOTAL_REQUESTS} 请求 × {EXCL_STRESS} 账号）')

    pool_name = f'stress_{TAG}'
    mgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=30)
    for i in range(EXCL_STRESS):
        aid = await mgr.add_account(pool_name, f's{i:03d}', credentials={'pwd': str(i)})
        created_ids.append(aid)

    # 每个 worker 独立记录结果，避免 set 并发竞态
    lock = asyncio.Lock()
    worker_results = []
    latencies = []
    errors = []

    async def worker(i: int):
        wmgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=30)
        t0 = time.perf_counter()
        try:
            act = await wmgr.acquire(pool_name, ttl_seconds=20)
            elapsed = time.perf_counter() - t0
            async with lock:
                latencies.append(elapsed)
                worker_results.append({'id': act.account_id, 'lease': act.lease_id})
            await asyncio.sleep(0.02)
            await wmgr.release(act.lease_id)
        except NoAvailableAccountError:
            elapsed = time.perf_counter() - t0
            async with lock:
                latencies.append(elapsed)
                worker_results.append({'id': None})
        except Exception as exc:
            async with lock:
                errors.append(f'worker={i}: {exc}')
        finally:
            await wmgr.aclose()

    t_start = time.perf_counter()
    await asyncio.gather(*[worker(i) for i in range(TOTAL_REQUESTS)])
    t_total = time.perf_counter() - t_start

    avg_lat = sum(latencies) / len(latencies) * 1000 if latencies else 0
    max_lat = max(latencies) * 1000 if latencies else 0
    min_lat = min(latencies) * 1000 if latencies else 0
    throughput = TOTAL_REQUESTS / t_total

    acquired_ids = {r['id'] for r in worker_results if r['id'] is not None}
    acquired_count = len(acquired_ids)
    no_avail_count = len([r for r in worker_results if r['id'] is None])

    print(f'  总耗时: {t_total:.2f}s')
    print(f'  吞吐: {throughput:.1f} req/s')
    print(f'  获取成功: {acquired_count}')
    print(f'  无可用账号: {no_avail_count}')
    print(f'  去重后 account_id: {len(acquired_ids)}')
    print(f'  延迟 avg={avg_lat:.1f}ms  min={min_lat:.1f}ms  max={max_lat:.1f}ms')
    if errors:
        for e in errors[:5]:
            print(f'  ⚠️  {e}')

    # 验证
    ok = len(acquired_ids) == EXCL_STRESS and not errors and acquired_count >= EXCL_STRESS
    if ok:
        PASS += 1
        print('  ✅ [PASS] E')
    else:
        FAIL += 1
        print('  ❌ [FAIL] E')

    # 清理
    mgr2 = AccountPoolManager(endpoint=ENDPOINT)
    try:
        await mgr2.release_all()
        for a in await mgr2.list_accounts(pool=pool_name):
            try:
                await mgr2.delete_account(a.id)
            except Exception:
                pass
    finally:
        await mgr2.aclose()
    # 顶部创建 pool 的 mgr 也要收尾
    await mgr.aclose()


# ============================================================================
# 测试 F：reserve 并发安全性
# ============================================================================


async def test_f_reserve_concurrency():
    """验证 reserve 不会被并发 acquire 抢走已预留的账号。"""
    global PASS, FAIL
    banner('F: reserve 并发安全性（reserve + 并发 acquire）')

    mgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=30)

    # reserve 5 个
    r = await mgr.reserve(EXCL_POOL, 5, ttl_seconds=120)
    print(f'  reserved {len(r.account_ids)} accounts: {r.account_ids}')

    # 10 个无 token 的去抢 —— 应该只能抢剩余 15 个（20-5=15）
    acquired_no_token = 0

    async def grab_no_token(i: int):
        nonlocal acquired_no_token
        wmgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=10)
        try:
            act = await wmgr.acquire(EXCL_POOL, ttl_seconds=10)
            acquired_no_token += 1
            # 验证：拿到的不能是 reserved 里的
            if act.account_id in r.account_ids:
                print(f'  ⚠️  BUG: 无 token 拿到了 reserved 账号 {act.account_id}!')
            await wmgr.release(act.lease_id)
        except NoAvailableAccountError:
            pass
        finally:
            await wmgr.aclose()

    # 5 个有 token 的去抢 —— 应该能拿到 reserved 里的
    acquired_with_token = 0

    async def grab_with_token(i: int):
        nonlocal acquired_with_token
        wmgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=10)
        try:
            act = await wmgr.acquire(EXCL_POOL, ttl_seconds=10, reservation_token=r.reservation_token)
            acquired_with_token += 1
            if act.account_id not in r.account_ids:
                print(f'  ⚠️  BUG: 有 token 拿到了非 reserved 账号 {act.account_id}')
            await wmgr.release(act.lease_id)
        except NoAvailableAccountError:
            pass
        finally:
            await wmgr.aclose()

    await asyncio.gather(
        *[grab_no_token(i) for i in range(10)],
        *[grab_with_token(i) for i in range(5)],
    )

    print(f'  无 token 获取: {acquired_no_token} (预期 ≤{EXCL_COUNT - 5})')
    print(f'  有 token 获取: {acquired_with_token} (预期 ≥1, ≤5)')

    ok = acquired_no_token <= EXCL_COUNT - 5 and acquired_with_token >= 1
    if ok:
        PASS += 1
        print('  ✅ [PASS] F')
    else:
        FAIL += 1
        print('  ❌ [FAIL] F')

    await mgr.cancel_reservation(r.reservation_token)
    await mgr.release_all()
    await mgr.aclose()


# ============================================================================
# 测试 G：共享池 reserve 并发（多批次同时预约同一共享池）
# ============================================================================


async def test_g_shared_reserve_concurrency():
    """8 批并发 reserve 同一共享池，每批 count=3；SHR_COUNT*SHR_CAP=50 容量足够。
    验证：
      1) 每批都能拿到 reservation_token 且 account_ids 长度=3
      2) 每批用 token acquire 都能翻牌成功（不报 no available account）
      3) 全部 release-to-reserved 后 shared_occupied 仍等于 sum(count)
      4) cancel 全部后 shared_occupied 回到 0
    """
    global PASS, FAIL
    banner('G: 共享池 reserve 并发（8 批 × 3 份，共 24/50）')

    BATCHES = 8
    PER = 3

    async def one_batch(i: int):
        m = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=60)
        try:
            r = await m.reserve(SHR_POOL, PER, ttl_seconds=120)
            assert len(r.account_ids) == PER, f'batch {i} 拿到 {len(r.account_ids)} 份'
            # 用 token 依次 acquire —— 每次都应该翻牌成功
            leases = []
            for _ in range(PER):
                act = await m.acquire(SHR_POOL, ttl_seconds=30, reservation_token=r.reservation_token)
                leases.append(act.lease_id)
            # release-to-reserved（默认）—— 应回到该批次的 reserved 占位
            for lid in leases:
                await m.release(lid)
            return {'ok': True, 'token': r.reservation_token, 'mgr': m}
        except Exception as exc:
            print(f'  ⚠️  batch {i} error: {exc}')
            await m.aclose()
            return {'ok': False, 'error': str(exc)}

    results = await asyncio.gather(*[one_batch(i) for i in range(BATCHES)])
    succ = [r for r in results if r['ok']]
    print(f'  成功批次: {len(succ)}/{BATCHES}')

    # 检查 shared_occupied
    m2 = AccountPoolManager(endpoint=ENDPOINT)
    stats = {s.pool: s for s in await m2.get_pool_stats()}
    s = stats[SHR_POOL]
    expected_occupied = len(succ) * PER
    print(f'  shared_occupied={s.shared_occupied} (预期 {expected_occupied})')

    ok1 = len(succ) == BATCHES and s.shared_occupied == expected_occupied

    # cancel 全部
    for r in succ:
        try:
            await r['mgr'].cancel_reservation(r['token'])
        except Exception as exc:
            print(f'  ⚠️  cancel {r["token"][:8]} error: {exc}')
        finally:
            await r['mgr'].aclose()

    stats2 = {s.pool: s for s in await m2.get_pool_stats()}
    s2 = stats2[SHR_POOL]
    await m2.aclose()
    print(f'  cancel 后 shared_occupied={s2.shared_occupied} (预期 0)')

    ok2 = s2.shared_occupied == 0
    if ok1 and ok2:
        PASS += 1
        print('  ✅ [PASS] G')
    else:
        FAIL += 1
        print('  ❌ [FAIL] G')


# ============================================================================
# 测试 H：release-to-reserved 并发 & 反复 acquire/release
# ============================================================================


async def test_h_release_to_reserved_loop():
    """一个 reservation 内多 worker 循环 acquire→release，验证账号能持续在同批次内翻牌复用。"""
    global PASS, FAIL
    banner('H: release-to-reserved 循环复用（4 worker × 5 轮 in 1 reservation）')

    WORKERS = 4
    ROUNDS = 5

    m = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=60)
    # 独占池 reserve 4 份
    r = await m.reserve(EXCL_POOL, WORKERS, ttl_seconds=180)
    assert len(r.account_ids) == WORKERS
    reserved_set = set(r.account_ids)

    ok = True
    lock = asyncio.Lock()

    async def worker(i: int):
        nonlocal ok
        wm = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=30)
        try:
            for rnd in range(ROUNDS):
                act = await wm.acquire(EXCL_POOL, ttl_seconds=20, reservation_token=r.reservation_token)
                async with lock:
                    if act.account_id not in reserved_set:
                        print(f'  ⚠️  worker {i} round {rnd} 拿到非 reserved 账号 {act.account_id}')
                        ok = False
                await asyncio.sleep(0.01)
                await wm.release(act.lease_id)  # release-to-reserved
        except Exception as exc:
            print(f'  ⚠️  worker {i} error: {exc}')
            ok = False
        finally:
            await wm.aclose()

    await asyncio.gather(*[worker(i) for i in range(WORKERS)])

    # 验证：这批账号 status 仍是 reserved，reservation_id 未变
    m2 = AccountPoolManager(endpoint=ENDPOINT)
    accts = await m2.list_accounts(pool=EXCL_POOL)
    reserved_now = {a.id for a in accts if a.status == 'reserved' and a.reservation_id == r.reservation_token}
    print(f'  仍处于 reserved 状态的账号: {len(reserved_now)} / {WORKERS}')
    ok = ok and reserved_now == reserved_set

    # 清理：cancel
    await m.cancel_reservation(r.reservation_token)
    await m.aclose()
    await m2.aclose()

    if ok:
        PASS += 1
        print('  ✅ [PASS] H')
    else:
        FAIL += 1
        print('  ❌ [FAIL] H')


# ============================================================================
# 测试 I：cancel_reservation 严格性并发
# ============================================================================


async def test_i_cancel_strictness():
    """一批 reserve 后并发 acquire，再从另一 worker 尝试 cancel —— 应被拒绝。
    随后 release 全部，再 cancel —— 应成功。
    """
    global PASS, FAIL
    banner('I: cancel_reservation 严格性（有 acquired 时拒绝，release 后放行）')

    m = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=60)
    r = await m.reserve(EXCL_POOL, 5, ttl_seconds=120)
    print('  reserved 5 accounts')

    # 3 个 worker 各自 acquire
    async def grab(i: int):
        wm = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=30)
        act = await wm.acquire(EXCL_POOL, ttl_seconds=30, reservation_token=r.reservation_token)
        return {'act': act, 'mgr': wm}

    grabs = await asyncio.gather(*[grab(i) for i in range(3)])
    print('  3 worker acquired, 2 剩余 reserved')

    # 并发尝试 cancel —— 每次都应被拒
    cancel_ok = 0
    cancel_rejected = 0

    async def try_cancel(i: int):
        nonlocal cancel_ok, cancel_rejected
        cm = AccountPoolManager(endpoint=ENDPOINT)
        try:
            await cm.cancel_reservation(r.reservation_token)
            cancel_ok += 1
            print(f'  ⚠️  cancel #{i} 意外成功')
        except Exception as exc:
            msg = str(exc)
            if 'acquired' in msg or 'release them first' in msg:
                cancel_rejected += 1
            else:
                print(f'  ⚠️  cancel #{i} 未预期错误: {exc}')
        finally:
            await cm.aclose()

    await asyncio.gather(*[try_cancel(i) for i in range(5)])
    print(f'  cancel 尝试：成功={cancel_ok}, 被拒={cancel_rejected}')

    ok1 = cancel_ok == 0 and cancel_rejected == 5

    # release 所有已 acquired
    for g in grabs:
        try:
            await g['mgr'].release(g['act'].lease_id)
        finally:
            await g['mgr'].aclose()

    # 再 cancel —— 应成功
    try:
        await m.cancel_reservation(r.reservation_token)
        cancel_after_release_ok = True
    except Exception as exc:
        print(f'  ⚠️  release 后 cancel 仍失败: {exc}')
        cancel_after_release_ok = False

    await m.aclose()

    if ok1 and cancel_after_release_ok:
        PASS += 1
        print('  ✅ [PASS] I')
    else:
        FAIL += 1
        print('  ❌ [FAIL] I')


# ============================================================================
# 测试 J：invalidate 与 acquire/release 并发
# ============================================================================


async def test_j_invalidate_race():
    """
    独占账号被一个 ctx 抢占并调用 invalidate；其他 ctx 同时尝试 acquire。
    验证：
      1) invalidate 成功后账号进入 invalid 状态
      2) 其他 ctx acquire 不会拿到 invalid 账号
      3) 已 acquired 的 lease 调用 release 不应崩溃
    """
    global PASS, FAIL
    banner('J: invalidate 与并发 acquire')

    m = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=60)
    # 加一个独立池，只 2 个账号，便于观察
    pool = f'inv_race_{TAG}'
    ids = []
    for i in range(2):
        aid = await m.add_account(pool, f'iv{i}', credentials={'pwd': str(i)})
        ids.append(aid)
        created_ids.append(aid)

    # ctx1: acquire 一个然后 invalidate
    ctx1 = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=30)
    act1 = await ctx1.acquire(pool, ttl_seconds=30)
    print(f'  ctx1 acquired id={act1.account_id}')

    invalidated_id = act1.account_id
    grabbers_result = []

    async def invalidator():
        # 稍等 grabbers 就绪
        await asyncio.sleep(0.05)
        await act1.invalidate(reason='race test')

    async def grabber(i: int):
        gm = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=30)
        try:
            act = await gm.acquire(pool, ttl_seconds=15)
            grabbers_result.append(act.account_id)
            await gm.release(act.lease_id)
        except NoAvailableAccountError:
            grabbers_result.append(None)
        except Exception as exc:
            grabbers_result.append(f'ERR:{exc}')
        finally:
            await gm.aclose()

    # 8 个 grabber + 1 invalidator 并发
    await asyncio.gather(
        invalidator(),
        *[grabber(i) for i in range(8)],
    )

    print(f'  grabber 结果: {grabbers_result}')
    got_invalid = any(x == invalidated_id for x in grabbers_result)
    err_count = sum(1 for x in grabbers_result if isinstance(x, str) and x.startswith('ERR'))

    # 验证账号状态
    accts = {a.id: a for a in await m.list_accounts(pool=pool)}
    inv_status = accts[invalidated_id].status
    print(f'  账号 {invalidated_id} 最终 status = {inv_status}')

    ok = (not got_invalid) and err_count == 0 and inv_status == 'invalid'

    # 恢复 + 清理
    try:
        await m.recover_account(invalidated_id)
    except Exception:
        pass
    await ctx1.aclose()
    await m.aclose()

    if ok:
        PASS += 1
        print('  ✅ [PASS] J')
    else:
        FAIL += 1
        print('  ❌ [FAIL] J')


# ============================================================================
# 测试 K：死锁探测（交叉多池预约）
# ============================================================================


async def test_k_deadlock_cross_pool():
    """
    两个 worker 交叉预约独占池 A、B：
      - w1: reserve(A, 1) → reserve(B, 1)
      - w2: reserve(B, 1) → reserve(A, 1)
    两池各只有 1 个账号，理论上可能"经典死锁"。
    通过 asyncio.wait_for 30s 超时兜底：如超时则失败。
    正确行为：因 server 端 reserve 是单次事务、行级锁按 pool 顺序取，不应死锁；
    但可能出现"insufficient accounts"错误——这属于业务错误，非死锁。
    """
    global PASS, FAIL
    banner('K: 死锁探测（跨池交叉预约，30s 超时兜底）')

    pool_a = f'dl_a_{TAG}'
    pool_b = f'dl_b_{TAG}'
    m = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=60)
    for p in (pool_a, pool_b):
        aid = await m.add_account(p, f'{p}_0', credentials={'pwd': 'x'})
        created_ids.append(aid)

    async def w1():
        wm = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=60)
        try:
            r1 = await wm.reserve(pool_a, 1, ttl_seconds=60)
            await asyncio.sleep(0.1)
            try:
                r2 = await wm.reserve(pool_b, 1, ttl_seconds=60)
                await wm.cancel_reservation(r2.reservation_token)
            except Exception as e:
                print(f'    w1: reserve B failed ({e}) - 业务错误可接受')
            await wm.cancel_reservation(r1.reservation_token)
            return 'ok'
        finally:
            await wm.aclose()

    async def w2():
        wm = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=60)
        try:
            r1 = await wm.reserve(pool_b, 1, ttl_seconds=60)
            await asyncio.sleep(0.1)
            try:
                r2 = await wm.reserve(pool_a, 1, ttl_seconds=60)
                await wm.cancel_reservation(r2.reservation_token)
            except Exception as e:
                print(f'    w2: reserve A failed ({e}) - 业务错误可接受')
            await wm.cancel_reservation(r1.reservation_token)
            return 'ok'
        finally:
            await wm.aclose()

    try:
        results = await asyncio.wait_for(
            asyncio.gather(w1(), w2(), return_exceptions=True),
            timeout=30,
        )
        print(f'  results: {results}')
        deadlock = False
    except asyncio.TimeoutError:
        deadlock = True
        print('  ⚠️  死锁：30s 内未完成！')

    # 用 multi_reservation 场景再测一次（同 token 原子预约多池）
    try:

        async def w_multi(items):
            wm = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=60)
            try:
                r = await wm.reserve_multi(items, ttl_seconds=60)
                await asyncio.sleep(0.05)
                await wm.cancel_reservation(r.reservation_token)
                return True
            except Exception as e:
                print(f'    multi reserve failed: {e}')
                return False
            finally:
                await wm.aclose()

        multi_results = await asyncio.wait_for(
            asyncio.gather(
                w_multi([{'pool': pool_a, 'count': 1}, {'pool': pool_b, 'count': 1}]),
                w_multi([{'pool': pool_b, 'count': 1}, {'pool': pool_a, 'count': 1}]),
                return_exceptions=True,
            ),
            timeout=30,
        )
        print(f'  multi_reserve results: {multi_results}')
        multi_deadlock = False
    except asyncio.TimeoutError:
        multi_deadlock = True
        print('  ⚠️  multi_reserve 死锁：30s 内未完成！')

    await m.aclose()

    if not deadlock and not multi_deadlock:
        PASS += 1
        print('  ✅ [PASS] K')
    else:
        FAIL += 1
        print('  ❌ [FAIL] K')


# ============================================================================
# 测试 L：心跳并发 & heartbeat_task 泄漏检测
# ============================================================================


async def test_l_heartbeat_no_leak():
    """
    在同一 ctx 内并发 acquire + 主动 release 循环若干轮，
    验证 _heartbeat_tasks / _accounts 无残留、无异常 Task 未 await。
    """
    global PASS, FAIL
    banner('L: 心跳 / heartbeat_task 泄漏检测（100 次 acquire→release）')

    mgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=8)  # 短 TTL 触发心跳

    max_ht = 0
    max_ac = 0
    errors = []

    async def cycle(i: int):
        nonlocal max_ht, max_ac
        try:
            act = await mgr.acquire(EXCL_POOL, ttl_seconds=8)
            max_ht = max(max_ht, len(mgr._heartbeat_tasks))
            max_ac = max(max_ac, len(mgr._accounts))
            await asyncio.sleep(0.02)
            await mgr.release(act.lease_id)
        except NoAvailableAccountError:
            pass
        except Exception as exc:
            errors.append(f'cycle {i}: {exc}')

    # 5 轮 × 20 并发
    for round_i in range(5):
        await asyncio.gather(*[cycle(i) for i in range(20)])

    print(f'  峰值 _heartbeat_tasks: {max_ht}')
    print(f'  峰值 _accounts: {max_ac}')
    print(f'  当前 _heartbeat_tasks: {len(mgr._heartbeat_tasks)}')
    print(f'  当前 _accounts: {len(mgr._accounts)}')
    if errors:
        for e in errors[:5]:
            print(f'  ⚠️  {e}')

    ok = (
        not errors and len(mgr._heartbeat_tasks) == 0 and len(mgr._accounts) == 0 and max_ht <= 20  # 峰值不应超过并发数
    )
    await mgr.aclose()

    if ok:
        PASS += 1
        print('  ✅ [PASS] L')
    else:
        FAIL += 1
        print('  ❌ [FAIL] L')


# ============================================================================
# main
# ============================================================================


async def main():
    global PASS, FAIL

    print('🔬 并发抢占评测开始')
    print(f'   Tag: {TAG}')
    print(f'   独占池: {EXCL_POOL} ({EXCL_COUNT} 账号)')
    print(f'   共享池: {SHR_POOL} ({SHR_COUNT} × cap={SHR_CAP})')

    # setup
    mgr = AccountPoolManager(endpoint=ENDPOINT, ttl_seconds=60)
    await setup(mgr)
    await mgr.aclose()

    try:
        await test_a_multi_ctx_concurrent()
        await test_b_single_ctx_internal_concurrency()
        await test_c_ctx_cleanup_concurrency()
        await test_d_mixed_concurrency()
        await test_e_stress()
        await test_f_reserve_concurrency()
        await test_g_shared_reserve_concurrency()
        await test_h_release_to_reserved_loop()
        await test_i_cancel_strictness()
        await test_j_invalidate_race()
        await test_k_deadlock_cross_pool()
        await test_l_heartbeat_no_leak()
    finally:
        # cleanup
        print(f'\n{"=" * 60}')
        print('🧹 清理...')
        cmgr = AccountPoolManager(endpoint=ENDPOINT)
        try:
            await cmgr.release_all()
            await asyncio.sleep(2)  # 等 reaper
            for aid in sorted(set(created_ids)):
                try:
                    await cmgr.delete_account(aid)
                except Exception:
                    pass
        finally:
            await cmgr.aclose()

    # summary
    print(f'\n{"=" * 60}')
    total = PASS + FAIL
    if FAIL == 0:
        print(f'🏆 全部通过！{PASS}/{total}')
    else:
        print(f'⚠️  {PASS} 通过, {FAIL} 失败 ({total} 总计)')
    print(f'{"=" * 60}')
    return FAIL == 0


if __name__ == '__main__':
    sys.exit(0 if asyncio.run(main()) else 1)
