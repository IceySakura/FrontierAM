from pydantic import BaseModel

from nexus.utils.aiofiles import read_file_or_empty, read_json_model_or_none


class SampleModel(BaseModel):
    name: str
    count: int


async def test_read_file_or_empty_returns_text(tmp_path):
    path = tmp_path / 'data.txt'
    path.write_text('hello', encoding='utf-8')

    assert await read_file_or_empty(str(path)) == 'hello'


async def test_read_file_or_empty_missing_returns_empty(tmp_path):
    assert await read_file_or_empty(str(tmp_path / 'missing.txt')) == ''


async def test_read_json_model_or_none_validates_model(tmp_path):
    path = tmp_path / 'data.json'
    path.write_text('{"name":"x","count":3}', encoding='utf-8')

    parsed = await read_json_model_or_none(str(path), SampleModel)

    assert parsed == SampleModel(name='x', count=3)


async def test_read_json_model_or_none_invalid_or_missing_returns_none(tmp_path):
    invalid = tmp_path / 'bad.json'
    invalid.write_text('{bad json', encoding='utf-8')
    wrong_shape = tmp_path / 'wrong.json'
    wrong_shape.write_text('{"name":"x"}', encoding='utf-8')

    assert await read_json_model_or_none(str(invalid), SampleModel) is None
    assert await read_json_model_or_none(str(wrong_shape), SampleModel) is None
    assert await read_json_model_or_none(str(tmp_path / 'missing.json'), SampleModel) is None
