"""Toolkit for the instore domain."""

from nexus.extensions.tasks.vita.data_model.tasks import Order
from nexus.extensions.tasks.vita.domains.instore.data_model import (
    BookInfo,
    InStoreDB,
    ReservationInfo,
    ShopProduct,
)
from nexus.extensions.tasks.vita.environment.toolkit import ToolKitBase, ToolType, is_tool
from nexus.extensions.tasks.vita.vita_utils.utils import check_time_format, rerank


class InStoreTools(ToolKitBase):
    """All the tools for the instore domain."""

    db: InStoreDB

    def __init__(self, db: InStoreDB) -> None:
        super().__init__(db)

    def _check_user(self, user_id: str) -> bool:
        """Check if the user is valid.
        Args:
            user_id: The user id
        Returns:
            bool: True if the user is valid, False otherwise
        """
        return user_id == self.db.user_id

    def _get_shop_tags(self) -> dict[str, str]:
        """Get the shop tags from the database.
        Returns:
            dict: {shop_id: shop_name + ',' + tags}
        """
        tag_dict = {}
        for shop in self.db.shops.values():
            tag_dict[shop.shop_id] = shop.shop_name + ',' + ','.join(shop.tags)
        return tag_dict

    def _get_shop_product_tags(self) -> dict[str, str]:
        """Get the product tags from the database.
        Returns:
            dict: {shop_id_product_id: shop_name shop_tags product_name product_tags}
        """
        product_tags = {}
        products = ShopProduct.get_all_products()
        for product_id, product in products.items():
            if isinstance(product, ShopProduct):
                product_tags[product_id] = product.name + ',' + ','.join(product.tags)

        return product_tags

    def _get_shop(self, shop_id: str):
        if shop_id not in self.db.shops:
            raise ValueError(f'Shop {shop_id} does not exist')
        return self.db.shops[shop_id]

    def _get_shop_product(self, product_id: str):
        product_dict = ShopProduct.get_all_products()
        if product_id not in product_dict:
            raise ValueError(f'Product {product_id} does not exist')
        product = product_dict[product_id]
        if not isinstance(product, ShopProduct):
            raise ValueError(f'Product {product_id} is not an instore scenario product')
        return product

    def _add_book_info(self, book_info: BookInfo) -> str:
        if book_info.book_id in self.db.books:
            return 'BookInfo already exists'
        self.db.books[book_info.book_id] = book_info
        return 'done'

    def _add_reservation_info(self, reservation_info: ReservationInfo) -> str:
        if reservation_info.reservation_id in self.db.reservations:
            return 'ReservationInfo already exists'
        self.db.reservations[reservation_info.reservation_id] = reservation_info
        return 'done'

    def _add_instore_order(self, order: Order) -> str:
        assert self.db.orders is not None
        if order.order_id in self.db.orders:
            return 'Order already exists'
        self.db.orders[order.order_id] = order
        return 'done'

    def _get_instore_order(self, order_id: str | None = None) -> Order | list[Order]:
        """Get the order from the database.

        Args:
            order_id: The order id

        Returns:
            The order.

        Raises:
            ValueError: If the order is not found.
        """
        assert self.db.orders is not None
        if order_id is None:
            return [order for order in self.db.orders.values() if order.order_type == 'instore']
        if order_id not in self.db.orders:
            raise ValueError(f'Order {order_id} not found')
        order = self.db.orders[order_id]
        if order.order_type != 'instore':
            raise ValueError(f'Order {order_id} is not an instore order')
        return order

    def _get_book_info(self, book_id: str | None = None) -> BookInfo | list[BookInfo]:
        """Get the book info from the database."""
        if book_id is None:
            return list(self.db.books.values())
        if book_id not in self.db.books:
            raise ValueError(f'BookInfo {book_id} not found')
        return self.db.books[book_id]

    def _get_reservation_info(self, reservation_id: str | None = None) -> ReservationInfo | list[ReservationInfo]:
        """Get the reservation info from the database."""
        if reservation_id is None:
            return list(self.db.reservations.values())
        if reservation_id not in self.db.reservations:
            raise ValueError(f'ReservationInfo {reservation_id} not found')
        return self.db.reservations[reservation_id]

    def _modify_instore_order(self, order: Order) -> str:
        """Modify order in domain-specific database.

        Args:
            order: The order to modify

        Returns:
            "done" if successful, error message otherwise
        """
        assert self.db.orders is not None
        if order.order_id not in self.db.orders:
            return f'Order {order.order_id} not found'

        self.db.orders[order.order_id] = order

        return 'done'

    def _modify_reservation_info(self, reservation_info: ReservationInfo) -> str:
        """Modify reservation info in domain-specific database.

        Args:
            reservation_info: The reservation info to modify

        Returns:
            "done" if successful, error message otherwise
        """
        if reservation_info.reservation_id not in self.db.reservations:
            return f'ReservationInfo {reservation_info.reservation_id} not found'

        self.db.reservations[reservation_info.reservation_id] = reservation_info

        return 'done'

    def _modify_book_info(self, book_info: BookInfo) -> str:
        """Modify book info in domain-specific database.

        Args:
            book_info: The book info to modify

        Returns:
            "done" if successful, error message otherwise
        """
        if book_info.book_id not in self.db.books:
            return f'BookInfo {book_info.book_id} not found'

        self.db.books[book_info.book_id] = book_info

        return 'done'

    @is_tool(tool_type=ToolType.READ)
    def instore_shop_search_recommend(self, keywords: list[str]) -> str:
        assert keywords, 'Keywords cannot be empty'
        assert isinstance(keywords, list), 'Keywords must be a list'
        assert all(isinstance(kw, str) and kw.strip() for kw in keywords), 'All keywords must be non-empty strings'

        top_k = 50
        shop_tag_dict = self._get_shop_tags()
        if not shop_tag_dict:
            return 'No shops available'

        try:
            keywords_str = ''.join(keywords)
            assert keywords_str, 'Keywords cannot be empty'
            assert keywords_str.strip(), 'Keywords cannot be empty'
            id_candidates_sorted = rerank(keywords_str, shop_tag_dict)
            selected_ids = [ic[0] for ic in id_candidates_sorted[:top_k]]
            if not selected_ids:
                return 'No shops found matching the keywords'

            selected_shops = []
            for shop_id in selected_ids:
                try:
                    selected_shops.append(str(self._get_shop(shop_id)))
                except ValueError:
                    continue

            if not selected_shops:
                return 'No shops found matching the keywords'

            return '\n'.join(selected_shops)
        except Exception as e:
            return f'Error searching shops: {e}'

    @is_tool(tool_type=ToolType.READ)
    def instore_product_search_recommend(self, keywords: list[str]) -> str:
        assert keywords, 'Keywords cannot be empty'
        assert isinstance(keywords, list), 'Keywords must be a list'
        assert all(isinstance(kw, str) and kw.strip() for kw in keywords), 'All keywords must be non-empty strings'

        top_k = 50
        product_tag_dict = self._get_shop_product_tags()
        if not product_tag_dict:
            return 'No products available'

        try:
            keywords_str = ''.join(keywords)
            assert keywords_str, 'Keywords cannot be empty'
            assert keywords_str.strip(), 'Keywords cannot be empty'
            id_candidates_sorted = rerank(keywords_str, product_tag_dict)
            selected_ids = [ic[0] for ic in id_candidates_sorted[:top_k]]
            if not selected_ids:
                return 'No products found matching the keywords'

            product_list = []
            for product_id in selected_ids:
                try:
                    product_list.append(self._get_shop_product(product_id))
                except ValueError:
                    continue

            if not product_list:
                return 'No products found matching the keywords'

            return '\n'.join([repr(product) for product in product_list])
        except Exception as e:
            return f'Error searching products: {e}'

    @is_tool(tool_type=ToolType.WRITE)
    def create_instore_product_order(self, user_id: str, shop_id: str, product_id: str, quantity: int = 1) -> str:
        assert user_id, 'User ID cannot be empty'
        assert self._check_user(user_id), 'User ID does not match'
        assert shop_id, 'Shop ID cannot be empty'
        assert product_id, 'Product ID cannot be empty'
        assert isinstance(quantity, int), 'Quantity must be an integer'
        assert quantity > 0, 'Quantity must be greater than 0'

        try:
            shop = self._get_shop(shop_id)
        except ValueError as e:
            return f'Error: {e}'

        if product_id not in [product.product_id for product in shop.products]:
            return f'Product {product_id} does not exist in shop {shop_id}'

        try:
            product = self._get_shop_product(product_id)
        except ValueError as e:
            return f'Error: {e}'

        product.quantity = quantity

        order = Order(
            order_id=self.db.assign_order_id('instore', user_id),
            order_type='instore',
            user_id=user_id,
            store_id=shop_id,
            total_price=quantity * product.price,
            create_time=self.get_now('%Y-%m-%d %H:%M:%S'),
            update_time=self.get_now('%Y-%m-%d %H:%M:%S'),
            status='unpaid',
            products=[product],
        )

        response = self._add_instore_order(order)
        if response == 'done':
            return repr(order)
        return f'Failed to create order: {response}'

    @is_tool(tool_type=ToolType.WRITE)
    def pay_instore_order(self, order_id: str) -> str:
        assert order_id, 'Order ID cannot be empty'

        try:
            order = self._get_instore_order(order_id)
        except ValueError as e:
            return f'Error: {e}'

        assert isinstance(order, Order)
        if order.status == 'unpaid':
            order.status = 'paid'
            order.update_time = self.get_now('%Y-%m-%d %H:%M:%S')
            resp = self._modify_instore_order(order)
            if resp == 'done':
                return 'Payment successful'
            return f'Payment failed: {resp}'
        return f'Order {order_id} is not in `unpaid` status. Current status: {order.status}'

    @is_tool(tool_type=ToolType.WRITE)
    def instore_cancel_order(self, order_id: str) -> str:
        assert order_id, 'Order ID cannot be empty'

        try:
            order = self._get_instore_order(order_id)
        except ValueError as e:
            return f'Error: {e}'

        assert isinstance(order, Order)
        if order.status in ['cancelled']:
            return f'Order {order.order_id} is already cancelled.'

        order.status = 'cancelled'
        order.update_time = self.get_now('%Y-%m-%d %H:%M:%S')
        resp = self._modify_instore_order(order)
        if resp == 'done':
            return f'Order {order.order_id} is cancelled.'
        return f'Cancellation failed: {resp}'

    @is_tool(tool_type=ToolType.WRITE)
    def instore_book(self, user_id: str, shop_id: str, time: str, customer_count: int = 1) -> str:
        assert user_id, 'User ID cannot be empty'
        assert self._check_user(user_id), 'User ID does not match'
        assert shop_id, 'Shop ID cannot be empty'
        assert time, 'Table booking time cannot be empty'
        assert isinstance(customer_count, int), 'Customer count must be an integer'
        assert customer_count > 0, 'Number of customers for table booking must be greater than 0'
        assert check_time_format(time, '%Y-%m-%d %H:%M:%S'), (
            'Table booking time format is incorrect, correct format is %Y-%m-%d %H:%M:%S'
        )

        try:
            shop = self._get_shop(shop_id)
        except ValueError as e:
            return f'Error: {e}'

        if not shop.enable_book:
            return f'Shop {shop_id} does not support table booking'

        book_price = shop.book_price
        status = 'unpaid' if book_price > 0 else 'paid'

        book_info = BookInfo(
            book_id=self.db.assign_order_id('instore_book', user_id),
            shop_id=shop_id,
            book_time=time,
            customer_count=customer_count,
            book_price=book_price,
            customer_id=user_id,
            status=status,
            update_time=self.get_now('%Y-%m-%d %H:%M:%S'),
        )

        response = self._add_book_info(book_info)
        if response == 'done':
            return repr(book_info)
        return f'Failed to create booking: {response}'

    @is_tool(tool_type=ToolType.WRITE)
    def pay_instore_book(self, book_id: str) -> str:
        assert book_id, 'Booking ID cannot be empty'

        try:
            book_info = self._get_book_info(book_id)
        except ValueError as e:
            return f'Error: {e}'

        assert isinstance(book_info, BookInfo)
        if book_info.status == 'unpaid':
            book_info.status = 'paid'
            book_info.update_time = self.get_now('%Y-%m-%d %H:%M:%S')
            resp = self._modify_book_info(book_info)
            if resp == 'done':
                return 'Payment successful'
            return f'Payment failed: {resp}'
        return f'BookInfo {book_info.book_id} is not in `unpaid` status. Current status: {book_info.status}'

    @is_tool(tool_type=ToolType.WRITE)
    def instore_cancel_book(self, book_id: str) -> str:
        assert book_id, 'Booking ID cannot be empty'

        try:
            book_info = self._get_book_info(book_id)
        except ValueError as e:
            return f'Error: {e}'

        assert isinstance(book_info, BookInfo)
        if book_info.status in ['cancelled']:
            return f'BookInfo {book_info.book_id} is already cancelled.'

        book_info.status = 'cancelled'
        book_info.update_time = self.get_now('%Y-%m-%d %H:%M:%S')
        resp = self._modify_book_info(book_info)
        if resp == 'done':
            return f'BookInfo {book_info.book_id} is cancelled.'
        return f'Cancellation failed: {resp}'

    @is_tool(tool_type=ToolType.WRITE)
    def instore_reservation(self, user_id: str, shop_id: str, time: str, customer_count: int = 1) -> str:
        assert user_id, 'User ID cannot be empty'
        assert self._check_user(user_id), 'User ID does not match'
        assert shop_id, 'Shop ID cannot be empty'
        assert time, 'Reservation time cannot be empty'
        assert isinstance(customer_count, int), 'Customer count must be an integer'
        assert customer_count > 0, 'Number of customers for reservation must be greater than 0'
        assert check_time_format(time, '%Y-%m-%d %H:%M:%S'), (
            'Reservation time format is incorrect, correct format is %Y-%m-%d %H:%M:%S'
        )

        try:
            self._get_shop(shop_id)
        except ValueError as e:
            return f'Error: {e}'

        reservation = ReservationInfo(
            reservation_id=self.db.assign_order_id('instore_reservation', user_id),
            shop_id=shop_id,
            reservation_time=time,
            customer_id=user_id,
            customer_count=customer_count,
            status='unconsumed',
            update_time=self.get_now('%Y-%m-%d %H:%M:%S'),
        )

        response = self._add_reservation_info(reservation)
        if response == 'done':
            return repr(reservation)
        return f'Failed to create reservation: {response}'

    @is_tool(tool_type=ToolType.WRITE)
    def instore_modify_reservation(self, reservation_id: str, time: str, customer_count: int = 0) -> str:
        assert reservation_id, 'Reservation ID cannot be empty'
        assert time, 'Reservation time cannot be empty'
        assert isinstance(customer_count, int), 'Customer count must be an integer'
        assert customer_count >= 0, 'Number of customers for reservation must be greater than or equal to 0'
        assert check_time_format(time, '%Y-%m-%d %H:%M:%S'), (
            'Reservation time format is incorrect, correct format is %Y-%m-%d %H:%M:%S'
        )

        try:
            reservation_info = self._get_reservation_info(reservation_id)
        except ValueError as e:
            return f'Error: {e}'

        assert isinstance(reservation_info, ReservationInfo)
        if reservation_info.status in ['consumed', 'cancelled']:
            return f'ReservationInfo {reservation_info.reservation_id} is already {reservation_info.status}.'

        reservation_info.reservation_time = time
        reservation_info.customer_count = customer_count
        reservation_info.update_time = self.get_now('%Y-%m-%d %H:%M:%S')
        resp = self._modify_reservation_info(reservation_info)
        if resp == 'done':
            return repr(reservation_info)
        return f'Modification failed: {resp}'

    @is_tool(tool_type=ToolType.WRITE)
    def instore_cancel_reservation(self, reservation_id: str) -> str:
        assert reservation_id, 'Reservation ID cannot be empty'

        try:
            reservation_info = self._get_reservation_info(reservation_id)
        except ValueError as e:
            return f'Error: {e}'

        assert isinstance(reservation_info, ReservationInfo)
        if reservation_info.status in ['cancelled']:
            return f'ReservationInfo {reservation_info.reservation_id} is already cancelled.'

        reservation_info.status = 'cancelled'
        reservation_info.update_time = self.get_now('%Y-%m-%d %H:%M:%S')
        resp = self._modify_reservation_info(reservation_info)
        if resp == 'done':
            return f'ReservationInfo {reservation_info.reservation_id} is cancelled.'
        return f'Cancellation failed: {resp}'

    @is_tool(tool_type=ToolType.READ)
    def get_instore_orders(self, user_id: str) -> str:
        assert user_id, 'User ID cannot be empty'
        assert self._check_user(user_id), 'User ID does not match'
        instore_orders_all = self._get_instore_order()
        assert isinstance(instore_orders_all, list)

        instore_orders = []
        for order in instore_orders_all:
            if order.user_id == user_id:
                instore_orders.append(order)

        if not instore_orders:
            return f'User {user_id} has no order information.'

        return '\n'.join([repr(order) for order in instore_orders])

    @is_tool(tool_type=ToolType.READ)
    def get_instore_reservations(self, user_id: str) -> str:
        assert user_id, 'User ID cannot be empty'
        assert self._check_user(user_id), 'User ID does not match'
        instore_reservations_all = self._get_reservation_info()
        assert isinstance(instore_reservations_all, list)
        instore_reservations = []
        for reservation in instore_reservations_all:
            if reservation.customer_id == user_id:
                instore_reservations.append(reservation)

        if not instore_reservations:
            return f'User {user_id} has no reservation information.'
        return '\n'.join([repr(reservation) for reservation in instore_reservations])

    @is_tool(tool_type=ToolType.READ)
    def get_instore_books(self, user_id: str) -> str:
        assert user_id, 'User ID cannot be empty'
        assert self._check_user(user_id), 'User ID does not match'
        instore_books_all = self._get_book_info()
        assert isinstance(instore_books_all, list)
        instore_books = []
        for book in instore_books_all:
            if book.customer_id == user_id:
                instore_books.append(book)

        if not instore_books:
            return f'User {user_id} has no book information.'
        return '\n'.join([repr(book) for book in instore_books])

    @is_tool(tool_type=ToolType.READ)
    def search_instore_book(self, user_id: str, book_id: str | None = None) -> str:
        assert user_id, 'User ID cannot be empty'
        assert self._check_user(user_id), 'User ID does not match'
        if book_id is None:
            instore_books_all = self._get_book_info()
            assert isinstance(instore_books_all, list)
            instore_books = []
            for book in instore_books_all:
                if book.customer_id == user_id:
                    instore_books.append(book)

            if not instore_books:
                return f'User {user_id} has no book information.'

            return '\n'.join([repr(book) for book in instore_books])
        book = self._get_book_info(book_id)
        assert isinstance(book, BookInfo)
        if book.customer_id != user_id:
            return f'BookInfo {book_id} is not belong to user {user_id}.'
        return repr(book)

    @is_tool(tool_type=ToolType.READ)
    def search_instore_reservation(self, user_id: str, reservation_id: str | None = None) -> str:
        assert user_id, 'User ID cannot be empty'
        assert self._check_user(user_id), 'User ID does not match'
        if reservation_id is None:
            user_reservations = []
            for reservation in self.db.reservations.values():
                if reservation.customer_id == user_id:
                    user_reservations.append(reservation)

            if not user_reservations:
                return f'User {user_id} has no reservation information.'

            return '\n'.join([repr(reservation) for reservation in user_reservations])
        reservation = self._get_reservation_info(reservation_id)
        assert isinstance(reservation, ReservationInfo)
        if reservation.customer_id != user_id:
            return f'ReservationInfo {reservation_id} is not belong to user {user_id}.'
        return repr(reservation)
