"""
通用 Image 实现，从 instance_dict 最外层读取配置信息
"""

from typing import Any

from nexus.extensions.tasks.swe.multi_swe_bench.image import Config, File, Image
from nexus.extensions.tasks.swe.multi_swe_bench.pull_request import PullRequest


class GenericImage(Image):
    """从 instance_dict 配置驱动的通用 Image 实现"""

    def __init__(self, pr: PullRequest, config: Config, instance_dict: dict[str, Any]):
        self._pr = pr
        self._config = config
        self._instance_dict = instance_dict

    @property
    def pr(self) -> PullRequest:
        return self._pr

    @property
    def config(self) -> Config:
        return self._config

    def dependency(self) -> str:
        """返回预构建的镜像名称"""
        return self._instance_dict.get('image', '')

    def image_tag(self) -> str:
        """从完整镜像名称中提取 tag"""
        image = self.dependency()
        if ':' in image:
            return image.split(':')[-1]
        return f'pr-{self.pr.number}'

    def workdir(self) -> str:
        """工作目录，优先从 instance_dict 读取，否则使用默认的 /home/{repo}"""
        return self._instance_dict.get('workdir', f'/home/{self.pr.repo}')

    def files(self) -> list[File]:
        """生成需要复制到镜像中的文件列表"""
        files = []
        scripts = self._instance_dict.get('scripts', {})

        # 添加各种脚本文件
        script_mapping = {
            'check_git_changes': 'check-git-changes.sh',
            'prepare': 'prepare.sh',
            'run': 'run.sh',
            'test_run': 'test-run.sh',
            'fix_run': 'fix-run.sh',
        }

        for config_key, filename in script_mapping.items():
            script_content = scripts.get(config_key)
            if script_content:
                files.append(File(dir='.', name=filename, content=script_content))

        return files

    def dockerfile(self) -> str:
        """生成 Dockerfile 内容"""
        # 使用预构建镜像，不需要生成 Dockerfile
        # 直接使用 dependency() 返回的镜像
        base_image = self.dependency()

        # 如果有文件需要复制，生成简单的 Dockerfile
        files = self.files()
        if files:
            copy_commands = ''
            for file in files:
                copy_commands += f'COPY {file.name} /home/\n'

            return f"""FROM {base_image}
{copy_commands}
WORKDIR {self.workdir()}
"""
        # 没有文件需要复制，直接使用基础镜像
        return f"""FROM {base_image}
WORKDIR {self.workdir()}
"""
