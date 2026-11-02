"""端到端测试：nexus.runtime.accounts 全部接口"""

import asyncio
import sys
import uuid


sys.path.insert(0, '/data/workspace/nexus')

from nexus.runtime.accounts import (
    AccountPoolManager,
    AsyncMultiReservation,
    AsyncReservation,
    NoAvailableAccountError,
)


TAG = uuid.uuid4().hex[:6]
EXCL_POOL = f'e2e_excl_{TAG}'
SHR_POOL = f'e2e_shr_{TAG}'
PASS = FAIL = 0
created_ids = []


async def main():
    global PASS, FAIL, created_ids
    mgr = AccountPoolManager(ttl_seconds=60)

    async def stats(pool):
        s = {s.pool: s for s in await mgr.get_pool_stats()}
        return s.get(pool)

    # ---- 1. add_account / list_accounts / get_pool_stats ----
    print('=== 1. add_account / list_accounts / get_pool_stats ===')
    try:
        ids = [
            await mgr.add_account(EXCL_POOL, f'u{i}', credentials={'pwd': f'p{i}'}, tokens={'tk': f't{i}'})
            for i in range(3)
        ]
        assert all(aid > 0 for aid in ids)
        created_ids.extend(ids)
        assert len(await mgr.list_accounts(pool=EXCL_POOL)) == 3
        s = await stats(EXCL_POOL)
        assert s.exclusive_total == 3 and s.exclusive_idle == 3
        PASS += 1
        print('[PASS] 1')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 1: {e}')

    # ---- 2. acquire / heartbeat / release ----
    print('\n=== 2. acquire / heartbeat / release ===')
    try:
        act = await mgr.acquire(EXCL_POOL, ttl_seconds=30)
        assert act.lease_id and act.account_id > 0
        assert 'pwd' in act.credentials
        assert await mgr.heartbeat(act.lease_id, ttl_seconds=60)
        await mgr.release(act.lease_id)
        assert (await stats(EXCL_POOL)).exclusive_idle == 3
        PASS += 1
        print('[PASS] 2')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 2: {e}')

    # ---- 3. exclusive contention (10 grab 3) ----
    print('\n=== 3. exclusive contention (10 grab 3) ===')
    try:
        ok = 0

        async def grab(_):
            nonlocal ok
            try:
                act = await mgr.acquire(EXCL_POOL, ttl_seconds=30)
                ok += 1
                await asyncio.sleep(0.1)
                await mgr.release(act.lease_id)
            except NoAvailableAccountError:
                pass

        await asyncio.gather(*[grab(i) for i in range(10)])
        await mgr.release_all()
        print(f'  acquired={ok}')
        PASS += 1
        print('[PASS] 3')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 3: {e}')

    # ---- 4. shared contention (cap=3, 5 grabbers) ----
    print('\n=== 4. shared contention (cap=3) ===')
    try:
        aid = await mgr.add_account(SHR_POOL, 'shared_bot', shared_cap=3)
        created_ids.append(aid)
        tasks = [mgr.acquire(SHR_POOL, ttl_seconds=30) for _ in range(5)]
        ok = no_avail = 0
        for t in asyncio.as_completed(tasks):
            try:
                await t
                ok += 1
            except NoAvailableAccountError:
                no_avail += 1
        print(f'  ok={ok} no_avail={no_avail}')
        assert ok == 3 and no_avail == 2
        await mgr.release_all()
        PASS += 1
        print('[PASS] 4')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 4: {e}')

    # ---- 5. reserve / acquire(token) / release-to-reserved / cancel ----
    print('\n=== 5. reserve / acquire(token) / release-to-reserved / cancel ===')
    try:
        result = await mgr.reserve(EXCL_POOL, 2, ttl_seconds=120)
        assert result.reservation_token and len(result.account_ids) == 2
        a1 = await mgr.acquire(EXCL_POOL, ttl_seconds=10, reservation_token=result.reservation_token)
        assert a1.account_id in result.account_ids
        # 使用 SDK 官方 release：因为携带了 reservation_token，
        # release 会自动把 token 带上，账号回到 reserved 而非 idle。
        await a1.release()
        s = await stats(EXCL_POOL)
        assert s.exclusive_reserved == 2 and s.exclusive_idle == 1
        await mgr.cancel_reservation(result.reservation_token)
        s = await stats(EXCL_POOL)
        assert s.exclusive_reserved == 0 and s.exclusive_idle == 3
        PASS += 1
        print('[PASS] 5')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 5: {e}')

    # ---- 6. reservation context manager ----
    print('\n=== 6. reservation async context manager ===')
    try:
        async with mgr.reservation(EXCL_POOL, 2, ttl_seconds=60, lease_ttl_seconds=10) as rsv:
            assert isinstance(rsv, AsyncReservation)
            async with rsv.acquire() as act:
                assert act.account_id in rsv.account_ids
        s = await stats(EXCL_POOL)
        assert s.exclusive_idle == 3 and s.exclusive_reserved == 0
        PASS += 1
        print('[PASS] 6')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 6: {e}')

    # ---- 7. multi_reservation context manager ----
    print('\n=== 7. multi_reservation async context manager ===')
    try:
        async with mgr.multi_reservation(
            [{'pool': EXCL_POOL, 'count': 2}], ttl_seconds=60, lease_ttl_seconds=10
        ) as rsv:
            assert isinstance(rsv, AsyncMultiReservation)
            assert rsv.pools == [EXCL_POOL]
            async with rsv.acquire(EXCL_POOL) as act:
                assert act.pool == EXCL_POOL
        s = await stats(EXCL_POOL)
        assert s.exclusive_idle == 3 and s.exclusive_reserved == 0
        PASS += 1
        print('[PASS] 7')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 7: {e}')

    # ---- 8. invalidate / recover ----
    print('\n=== 8. invalidate / recover ===')
    try:
        aid = (await mgr.list_accounts(pool=EXCL_POOL))[0].id
        await mgr.invalidate(aid)
        assert (await stats(EXCL_POOL)).exclusive_invalid >= 1
        await mgr.recover(aid)
        assert (await stats(EXCL_POOL)).exclusive_invalid == 0
        PASS += 1
        print('[PASS] 8')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 8: {e}')

    # ---- 9. delete_account ----
    print('\n=== 9. delete_account ===')
    try:
        aid = await mgr.add_account(EXCL_POOL, 'to_del', credentials={'x': 'y'})
        created_ids.append(aid)
        await mgr.delete_account(aid)
        created_ids.remove(aid)
        assert all(a.id != aid for a in await mgr.list_accounts(pool=EXCL_POOL))
        PASS += 1
        print('[PASS] 9')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 9: {e}')

    # ---- 10. AcquiredAccount.invalidate() ----
    print('\n=== 10. AcquiredAccount.invalidate() ===')
    inv_id = None
    try:
        act = await mgr.acquire(EXCL_POOL, ttl_seconds=30)
        inv_id = act.account_id
        await act.invalidate('test invalidation')
        assert act.invalidated
        PASS += 1
        print('[PASS] 10')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 10: {e}')

    # 恢复 case 10 里 invalidate 的账号，让 pool 回到 3 idle，便于后续用例。
    if inv_id is not None:
        try:
            await mgr.recover(inv_id)
        except Exception:
            pass

    # ---- 11. AsyncReservation: release-to-reserved 默认语义 ----
    # 覆盖修复：AsyncReservation.acquire() 内的 release 必须携带 reservation_token，
    # 保证退出 with 后账号回到 reserved 而不是 idle。
    print('\n=== 11. reservation.acquire() -> release keeps reserved ===')
    try:
        async with mgr.reservation(EXCL_POOL, 2, ttl_seconds=120, lease_ttl_seconds=10) as rsv:
            async with rsv.acquire() as act:
                assert act.account_id in rsv.account_ids
                # AcquiredAccount 内部字段应被正确填充：
                # 携带 reservation_token 后 release 自动回 reserved。
                assert act._reservation_token == rsv.token
            # 退出 rsv.acquire() 之后：账号已 release，但应仍在 reserved 里
            s = await stats(EXCL_POOL)
            assert s.exclusive_reserved == 2, f'expect 2 reserved after inner release, got {s.exclusive_reserved}'
            assert s.exclusive_idle == 1, f'expect 1 idle after inner release, got {s.exclusive_idle}'
        # 退出 mgr.reservation 之后：预留取消，全部回到 idle
        s = await stats(EXCL_POOL)
        assert s.exclusive_reserved == 0 and s.exclusive_idle == 3
        PASS += 1
        print('[PASS] 11')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 11: {e}')

    # ---- 12. acquire with reservation_token -> release keeps reserved ----
    # 覆盖修复：手工用 reserve + acquire(reservation_token=…) + release()
    # 因为携带了 token，应把账号还回 reserved 而不是 idle。
    print('\n=== 12. acquire(reservation_token) auto-releases to reserved ===')
    try:
        result = await mgr.reserve(EXCL_POOL, 2, ttl_seconds=120)
        a = await mgr.acquire(EXCL_POOL, ttl_seconds=10, reservation_token=result.reservation_token)
        assert a._reservation_token == result.reservation_token
        await a.release()
        s = await stats(EXCL_POOL)
        assert s.exclusive_reserved == 2, f'expect 2 reserved after release, got {s.exclusive_reserved}'
        assert s.exclusive_idle == 1
        # 收尾
        await mgr.cancel_reservation(result.reservation_token)
        s = await stats(EXCL_POOL)
        assert s.exclusive_reserved == 0 and s.exclusive_idle == 3
        PASS += 1
        print('[PASS] 12')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 12: {e}')

    # ---- 13. shared pool: reserve -> acquire(token) -> release-to-reserved -> cancel ----
    # 覆盖新功能：共享账号池也可以被 reserve，占位以 account_leases.state='reserved' 存在。
    # 预约期间 shared_occupied 计入预约槽位。
    print('\n=== 13. shared reserve/acquire/release/cancel full loop ===')
    try:
        # 用一个全新共享 pool，避免和 case 4 的 SHR_POOL 干扰
        shr2 = f'e2e_shr2_{TAG}'
        aid = await mgr.add_account(shr2, 'shared_bot2', shared_cap=3)
        created_ids.append(aid)

        def _acct():
            # 同步读取共享账号当前状态
            async def inner():
                accts = await mgr.list_accounts(pool=shr2)
                return next(a for a in accts if a.id == aid)

            return inner()

        # 预约 2 份
        result = await mgr.reserve(shr2, 2, ttl_seconds=120)
        assert result.reservation_token and len(result.account_ids) == 2
        # 一个共享账号被预约 2 份 -> account_ids 应该都指向同一个 aid
        assert all(x == aid for x in result.account_ids), (
            f'expect all share the same aid={aid}, got {result.account_ids}'
        )

        a = await _acct()
        assert a.shared_occupied == 2, f'after reserve: expect occupied=2, got {a.shared_occupied}'
        assert a.status == 'reserved', f'after reserve: expect status=reserved, got {a.status}'

        # 用 token acquire —— 翻牌其中一个 reserved 占位
        act = await mgr.acquire(shr2, ttl_seconds=30, reservation_token=result.reservation_token)
        assert act.account_id == aid
        assert act._reservation_token == result.reservation_token
        # 翻牌不改变占用槽位数
        a = await _acct()
        assert a.shared_occupied == 2, f'after acquire(token): expect occupied=2, got {a.shared_occupied}'

        # 携带 token 的 release -> 应回到 reserved，不释放槽位
        await act.release()
        a = await _acct()
        assert a.shared_occupied == 2, f'after release-to-reserved: expect occupied=2, got {a.shared_occupied}'
        assert a.status == 'reserved', f'after release-to-reserved: expect status=reserved, got {a.status}'

        # cancel_reservation 之后完全释放
        await mgr.cancel_reservation(result.reservation_token)
        a = await _acct()
        assert a.shared_occupied == 0, f'after cancel: expect occupied=0, got {a.shared_occupied}'
        assert a.status == 'idle', f'after cancel: expect status=idle, got {a.status}'
        PASS += 1
        print('[PASS] 13')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 13: {e}')

    # ---- 14. cancel_reservation refuses when batch has acquired lease ----
    # 覆盖修复：批次内还有 acquired 时 cancel 应报错，避免误杀活跃租约。
    print('\n=== 14. cancel_reservation rejects when acquired exists ===')
    try:
        result = await mgr.reserve(EXCL_POOL, 2, ttl_seconds=120)
        act = await mgr.acquire(EXCL_POOL, ttl_seconds=30, reservation_token=result.reservation_token)
        # 此时批次内有 1 个 acquired -> cancel 应报错
        rejected = False
        try:
            await mgr.cancel_reservation(result.reservation_token)
        except Exception as e:
            rejected = True
            print(f'  cancel rejected as expected: {e}')
        assert rejected, 'cancel_reservation should reject when acquired exists'

        # 校验此时 pool 状态未被破坏：1 acquired + 1 reserved + 1 idle
        s = await stats(EXCL_POOL)
        assert s.exclusive_acquired == 1 and s.exclusive_reserved == 1 and s.exclusive_idle == 1, (
            f'unexpected stats: acquired={s.exclusive_acquired} reserved={s.exclusive_reserved} idle={s.exclusive_idle}'
        )

        # 释放后 cancel 应成功，全部回 idle
        await act.release()  # 携带 token -> 回 reserved
        await mgr.cancel_reservation(result.reservation_token)
        s = await stats(EXCL_POOL)
        assert s.exclusive_reserved == 0 and s.exclusive_idle == 3
        PASS += 1
        print('[PASS] 14')
    except Exception as e:
        FAIL += 1
        print(f'[FAIL] 14: {e}')

    # ---- cleanup ----
    print('\n--- cleanup ---')
    await mgr.release_all()
    for aid in sorted(set(created_ids)):
        try:
            await mgr.delete_account(aid)
        except Exception:
            pass
    await mgr.aclose()

    print(f'\n{"=" * 40}\n结果: {PASS} 通过, {FAIL} 失败\n{"=" * 40}')
    return FAIL == 0


if __name__ == '__main__':
    sys.exit(0 if asyncio.run(main()) else 1)
