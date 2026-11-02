from typing import Any

from pydantic import Field, field_validator

from nexus.extensions.tasks.vita.data_model.tasks import Location, ProductBaseModel, StoreBaseModel
from nexus.extensions.tasks.vita.environment.db import DB


class StoreProduct(ProductBaseModel):
    """Represents a product with its variants"""

    name: str = Field(description='Name of the product')
    store_id: str = Field(description='ID of the store this product belongs to')
    store_name: str = Field(description='Name of the store')
    attributes: str | list = Field(description='Attributes of the product')
    tags: list[str] = Field(description='Tags of the product')

    def __repr__(self):
        return (
            f'StoreProduct(store_name={self.store_name}, '
            f'store_id={self.store_id}, '
            f'product_name={self.name}, '
            f'product_id={self.product_id}, '
            f'attributes={self.attributes}, '
            f'quantity={self.quantity}, '
            f'price={self.price}, '
            f'tags={self.tags})'
        )

    @field_validator('attributes')
    @classmethod
    def validate_attributes(cls, attributes: Any):
        if isinstance(attributes, list):
            return ', '.join(attributes)
        return attributes


class Store(StoreBaseModel):
    """Represents a store with its variants"""

    name: str = Field(description='Name of the store')
    score: float = Field(description='Score of the store')
    location: Location = Field(description='location of the store')
    tags: list[str] = Field(description='Tags of the store')
    products: list[StoreProduct] = Field(description='List of products')

    def __repr__(self):
        products_repr = '\n'.join(repr(p) for p in self.products)
        return (
            f'Store(name={self.name}, '
            f'store_id={self.store_id}, '
            f'score={self.score}, '
            f'location={repr(self.location)}, '
            f'tags={self.tags}), '
            f'products={products_repr}'
        )

    def __str__(self):
        return (
            f'Store(name={self.name}, '
            f'store_id={self.store_id}, '
            f'score={self.score}, '
            f'location={repr(self.location)}, '
            f'tags={self.tags})'
        )


class DeliveryDB(DB):
    """Database containing all delivery-related data including stores"""

    stores: dict[str, Store] = Field(description='Dictionary of all products indexed by product ID')

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        num_stores = len(self.stores)
        return {
            'num_stores': num_stores,
        }
