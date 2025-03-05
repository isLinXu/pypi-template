# Python PyPI 库模板

这是一个用于创建可发布到PyPI的Python库的模板项目。它提供了一个完整的项目结构和配置，使您能够快速开始开发自己的Python库。

本模板旨在解决Python库开发中的常见问题，提供标准化的项目结构和最佳实践，帮助开发者专注于核心功能的实现，而不是项目配置和结构设计。无论您是开发工具库、数据处理包还是Web框架，本模板都能为您提供坚实的基础。

> **🔰 新手友好提示**：如果您是第一次创建Python库，不用担心！本文档提供了详细的保姆级指南，帮助您一步步完成从模板到成品库的转换过程。

## 特性

- 完整的项目结构和配置
- 集成测试框架（pytest）
- 代码质量工具（black, isort, flake8, mypy）
- 自动化CI/CD流程（GitHub Actions）
- 完整的文档支持（Sphinx）
- 类型提示支持
- 开发工具集成（pre-commit, tox）

## 快速开始

### 使用这个模板

1. 点击GitHub上的"Use this template"按钮创建一个新的仓库
2. 克隆您的新仓库到本地
   ```bash
   git clone https://github.com/您的用户名/您的仓库名.git
   cd 您的仓库名
   ```
3. 按照下面的指南自定义您的项目

### 安装开发环境

```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate

# 安装开发依赖
pip install -e ".[dev]"
```

### 项目结构

```
.
├── LICENSE                 # Apache 2.0 许可证
├── README.md               # 项目说明文档
├── pyproject.toml          # 项目配置文件
├── src/                    # 源代码目录
│   └── example_package/    # 包目录（重命名为您的包名）
│       ├── __init__.py     # 包初始化文件
│       └── ...             # 其他模块
└── tests/                  # 测试目录
    └── ...                 # 测试文件
```

## 将模板转换为您的项目

将此模板转换为您自己的项目需要几个关键步骤。以下是一个完整的转换流程，即使您是编程新手也能轻松完成：

### 第1步：项目初始化

1. **使用模板创建新仓库**
   - 访问GitHub上的模板仓库页面
   - 点击页面上方的绿色按钮"Use this template"（使用此模板）
   - 填写您的新仓库名称（建议使用您计划的包名）
   - 选择公开或私有仓库
   - 点击"Create repository from template"（从模板创建仓库）

2. **克隆新仓库到本地**
   ```bash
   git clone https://github.com/您的用户名/您的仓库名.git
   cd 您的仓库名
   ```

3. **创建并激活虚拟环境**
   - 这一步非常重要，它可以隔离项目依赖，避免与系统其他Python项目冲突
   ```bash
   # 在项目根目录下创建虚拟环境
   python -m venv venv
   
   # 在macOS/Linux上激活虚拟环境
   source venv/bin/activate
   
   # 在Windows上激活虚拟环境
   venv\Scripts\activate
   ```
   
   激活成功后，您的命令行前面会出现`(venv)`前缀

4. **安装开发依赖**
   ```bash
   # 安装项目及其开发依赖
   pip install -e ".[dev]"
   ```
   
   这条命令会以可编辑模式安装您的包，同时安装所有开发所需的依赖项

5. **初始化Git钩子**（可选但强烈推荐）
   ```bash
   pre-commit install
   ```
   
   这会在每次提交代码前自动运行代码格式化和检查，确保代码质量

### 第2步：基础配置

1. **在`pyproject.toml`中修改项目元数据**：
   - 打开`pyproject.toml`文件，这是项目的核心配置文件
   - 修改以下关键字段：
     - `name`: 您的包名（确保在PyPI上是唯一的）
     - `version`: 版本号（遵循[语义化版本规范](https://semver.org/lang/zh-CN/)）
     - `description`: 项目简短描述（50-100字为宜）
     - `authors`: 作者信息，包括姓名和电子邮件
     - `classifiers`: 根据您的项目特点选择合适的分类（可在[PyPI分类列表](https://pypi.org/classifiers/)查看）
     - `dependencies`: 项目依赖项（列出您的包运行所需的所有外部包）
     - `optional-dependencies`: 可选依赖项（按功能分组）
     - `project.urls`: 更新项目相关链接，如主页、文档、源码仓库等
   
   修改前后对比示例：
   ```toml
   # 修改前
   [project]
   name = "example_package"
   version = "0.1.0"
   description = "A template package for PyPI distribution"
   authors = [
       {name = "Your Name", email = "your.email@example.com"}
   ]
   
   # 修改后
   [project]
   name = "your_package_name"
   version = "0.1.0"
   description = "您的项目描述：简洁明了地说明包的功能和用途"
   authors = [
       {name = "您的姓名", email = "您的邮箱@example.com"}
   ]
   ```
   
   同样更新项目URL：
   ```toml
   [project.urls]
   "Homepage" = "https://github.com/您的用户名/您的仓库名"
   "Bug Tracker" = "https://github.com/您的用户名/您的仓库名/issues"
   "Documentation" = "https://您的仓库名.readthedocs.io/"
   ```

2. **重命名包目录**：
   - 将示例包目录重命名为您的包名：
   ```bash
   # 在项目根目录下执行
   mv src/example_package src/your_package_name
   ```
   
   - 然后更新所有导入语句和引用。可以使用以下命令查找需要更新的地方：
   ```bash
   # 查找所有包含example_package的文件
   grep -r "example_package" .
   ```
   
   - 您需要修改的文件通常包括：
     - `tests/test_example.py`中的导入语句
     - `.github/workflows/ci.yml`中的测试路径
     - 任何其他引用了原包名的文件
   
   - 例如，在测试文件中：
   ```python
   # 修改前
   from example_package import __version__
   from example_package.core import DataPoint, ExampleClass
   
   # 修改后
   from your_package_name import __version__
   from your_package_name.core import DataPoint, ExampleClass
   ```

3. **更新包的初始化文件**：
   - 编辑`src/your_package_name/__init__.py`文件：
     - 更新文档字符串，清晰描述包的用途
     - 保持或修改版本号
     - 导入并暴露您希望用户可以直接访问的类和函数
     - 配置`__all__`列表，明确指定公开API
   
   ```python
   """您的包描述。

   详细说明包的用途、主要功能和使用场景。
   可以包含简短的示例代码。
   """

   __version__ = "0.1.0"
   
   # 导入您希望用户可以直接访问的类和函数
   from .core import ExampleClass, utility_function
   
   # 明确指定公开的API
   __all__ = ["ExampleClass", "utility_function"]
   ```

4. **更新许可证信息**：
   - 修改`LICENSE`文件：
     - 更新版权声明中的年份和所有者信息
     - 例如：`Copyright (c) 2023 您的姓名或组织`
   
   - 如果需要更换许可证类型：
     - 可以选择其他常见的开源许可证，如MIT（更宽松）或GPL（更严格）
     - 在[choosealicense.com](https://choosealicense.com/)选择合适的许可证
     - 替换整个LICENSE文件内容
     - 同时更新`pyproject.toml`中的license字段：
       ```toml
       license = {text = "MIT"} # 或其他许可证
       ```

5. **更新文档**：
   - 修改`README.md`文件：
     - 更新项目标题和描述
     - 添加安装说明（如`pip install your_package_name`）
     - 提供基本的使用示例代码
     - 说明主要功能和特性
     - 添加贡献指南和联系方式
   
   - 创建更详细的文档（可选但推荐）：
     - 创建`docs/`目录并使用Sphinx设置文档框架
     - 为每个模块、类和函数编写详细的文档字符串
     - 添加教程和高级用法示例
   
   - 创建`CHANGELOG.md`文件记录版本变更：
     ```markdown
     # 更新日志
     
     ## 0.1.0 (2023-XX-XX)
     
     - 初始版本发布
     - 实现了核心功能X
     - 添加了Y特性
     ```

### 第3步：更新测试和CI配置

1. 修改测试文件：
   - 将`tests/test_example.py`中的导入语句更新为您的包名
   - 根据您的实际功能调整测试用例
   - 添加新的测试文件覆盖所有核心功能

2. 更新CI配置：
   - 修改`.github/workflows/ci.yml`中的包名和测试路径
   - 根据需要调整Python版本支持范围
   - 配置发布流程的凭证和触发条件

### 进阶定制

1. 调整项目结构：
   - 根据功能模块组织代码结构
   - 添加新的子包和模块
   - 创建必要的资源文件目录

2. 配置开发工具：
   - 在`pyproject.toml`中自定义代码格式化规则
   - 调整类型检查器配置
   - 配置测试覆盖率要求

3. 持续集成设置：
   - 修改`.github/workflows/ci.yml`以满足特定需求
   - 添加自定义的CI/CD步骤
   - 配置自动发布流程

4. 添加新功能：
   - 实现核心功能模块
   - 编写单元测试
   - 添加集成测试
   - 更新文档和示例

### 最佳实践

1. 代码组织：
   - 保持模块职责单一
   - 使用清晰的命名约定
   - 添加适当的类型注解
   - 编写详细的文档字符串

2. 测试策略：
   - 单元测试覆盖核心功能
   - 添加集成测试用例
   - 包含性能测试（如需要）
   - 使用参数化测试提高覆盖率

3. 文档维护：
   - 保持README.md更新
   - 编写详细的API文档
   - 提供使用示例和教程
   - 记录重要的更改日志

4. 版本控制：
   - 遵循语义化版本规范
   - 维护更新日志
   - 使用Git标签标记发布版本
   - 创建发布说明

## 开发工作流

### 运行测试

```bash
# 运行所有测试
pytest

# 运行带覆盖率报告的测试
pytest --cov=src/your_package_name

# 运行特定测试文件
pytest tests/test_specific.py
```

### 代码格式化和检查

```bash
# 格式化代码
black src tests
isort src tests

# 代码检查
flake8 src tests
mypy src

# 使用pre-commit钩子（推荐）
pre-commit install  # 首次设置
pre-commit run --all-files  # 手动运行所有检查
```

### 构建分发包

```bash
# 安装构建工具（如果尚未安装）
pip install build

# 构建分发包
python -m build
```

### 发布到PyPI

```bash
# 安装twine（如果尚未安装）
pip install twine

# 检查分发包
twine check dist/*

# 发布到TestPyPI（测试）
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# 发布到PyPI
twine upload dist/*
```

### 版本发布流程

1. 更新版本号（在`__init__.py`和`pyproject.toml`中）
2. 更新CHANGELOG.md
3. 提交更改并创建版本标签：
   ```bash
   git add .
   git commit -m "Release vX.Y.Z"
   git tag vX.Y.Z
   git push origin main --tags
   ```
4. 构建并发布到PyPI

## 项目扩展指南

成功转换模板后，您可以根据项目需求进行扩展。以下是一些常见的扩展场景和实现方法：

### 添加命令行接口

如果您的包需要命令行功能，可以使用`click`或`argparse`库：

1. 添加依赖：
   ```toml
   # 在pyproject.toml中
   dependencies = [
       "click>=8.0",
   ]
   ```

2. 创建CLI模块：
   ```python
   # src/your_package_name/cli.py
   import click
   from . import core

   @click.command()
   @click.argument("input_value", type=int)
   @click.option("--factor", "-f", default=1.0, help="乘数因子")
   def main(input_value, factor):
       """示例命令行工具。"""
       result = core.utility_function(input_value, factor)
       click.echo(f"结果: {result}")

   if __name__ == "__main__":
       main()
   ```

3. 在`pyproject.toml`中注册入口点：
   ```toml
   [project.scripts]
   your-command = "your_package_name.cli:main"
   ```

### 添加插件系统

为您的包添加可扩展的插件系统：

1. 定义插件接口：
   ```python
   # src/your_package_name/plugin.py
   from abc import ABC, abstractmethod

   class PluginInterface(ABC):
       @abstractmethod
       def process(self, data):
           """处理数据的插件方法。"""
           pass
   ```

2. 使用入口点机制注册插件：
   ```toml
   # 在pyproject.toml中
   [project.entry-points."your_package_name.plugins"]
   default = "your_package_name.default_plugin:DefaultPlugin"
   ```

### 集成Web框架

如果您的包需要Web功能：

1. 添加依赖：
   ```toml
   # 在pyproject.toml的optional-dependencies中
   web = [
       "flask>=2.0",
       "gunicorn>=20.0",
   ]
   ```

2. 创建Web模块：
   ```python
   # src/your_package_name/web.py
   from flask import Flask, jsonify
   from . import core

   app = Flask(__name__)

   @app.route("/api/process/<int:value>")
   def process(value):
       result = core.utility_function(value)
       return jsonify({"result": result})

   def create_app():
       return app
   ```

### 添加异步支持

为您的包添加异步功能：

1. 创建异步模块：
   ```python
   # src/your_package_name/async_core.py
   import asyncio
   from typing import Any, Dict, List

   async def async_process_data(data: List[Any]) -> Dict[str, Any]:
       """异步处理数据的示例函数。"""
       # 模拟异步操作
       await asyncio.sleep(1)  # 模拟耗时操作
       return {
           "processed": True,
           "items_count": len(data),
           "timestamp": asyncio.get_event_loop().time()
       }

   async def batch_process(batch_data: List[List[Any]]) -> List[Dict[str, Any]]:
       """批量异步处理多组数据。"""
       tasks = [async_process_data(data) for data in batch_data]
       return await asyncio.gather(*tasks)
   ```

2. 在主模块中提供异步接口：
   ```python
   # src/your_package_name/__init__.py 中添加
   from .async_core import async_process_data, batch_process
   
   __all__ += ["async_process_data", "batch_process"]
   ```

### 添加数据处理功能

如果您的包需要处理数据：

1. 添加数据处理依赖：
   ```toml
   # 在pyproject.toml的optional-dependencies中
   data = [
       "numpy>=1.20",
       "pandas>=1.3",
       "scikit-learn>=1.0",
   ]
   ```

2. 创建数据处理模块：
   ```python
   # src/your_package_name/data_processing.py
   import numpy as np
   import pandas as pd
   from typing import Dict, List, Union, Optional

   def load_data(file_path: str) -> pd.DataFrame:
       """加载数据文件到DataFrame。"""
       if file_path.endswith('.csv'):
           return pd.read_csv(file_path)
       elif file_path.endswith('.json'):
           return pd.read_json(file_path)
       else:
           raise ValueError(f"不支持的文件格式: {file_path}")

   def preprocess_data(df: pd.DataFrame, options: Optional[Dict] = None) -> pd.DataFrame:
       """预处理数据。"""
       options = options or {}
       result = df.copy()
       
       # 处理缺失值
       if options.get('fill_na'):
           result = result.fillna(options['fill_na'])
       
       # 标准化数值列
       if options.get('normalize', False):
           for col in result.select_dtypes(include=[np.number]).columns:
               result[col] = (result[col] - result[col].mean()) / result[col].std()
               
       return result
   ```

### 添加国际化支持

为您的包添加多语言支持：

1. 创建本地化资源目录：
   ```bash
   mkdir -p src/your_package_name/locales/{zh_CN,en_US}/LC_MESSAGES
   ```

2. 使用gettext框架：
   ```python
   # src/your_package_name/i18n.py
   import gettext
   import os
   from typing import Optional

   def setup_i18n(locale: str = 'en_US') -> gettext.GNUTranslations:
       """设置国际化支持。"""
       localedir = os.path.join(os.path.dirname(__file__), 'locales')
       return gettext.translation('messages', localedir, [locale], fallback=True)

   # 默认使用英语
   _ = setup_i18n().gettext
   ```

## 贡献指南

欢迎贡献！请按照以下步骤参与项目：

1. Fork本仓库
2. 创建您的特性分支：`git checkout -b feature/amazing-feature`
3. 提交您的更改：`git commit -m 'Add some amazing feature'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 提交拉取请求

请确保您的代码通过所有测试并符合项目的代码风格。

## 许可证

本项目采用Apache 2.0许可证。详情请参阅[LICENSE](LICENSE)文件。

## 常见问题解答

### 如何添加新的依赖项？

在`pyproject.toml`文件的`dependencies`部分添加新的依赖项。如果是可选依赖，则添加到`optional-dependencies`部分。

```toml
# 添加必需依赖
[project]
dependencies = [
    "requests>=2.25.0",
    "numpy>=1.20.0",
]
```

### 如何确保我的包名在PyPI上是唯一的？

在创建包之前，您可以在[PyPI网站](https://pypi.org)上搜索您计划使用的名称，或者使用以下命令检查：

```bash
pip search 您的包名
```

如果搜索结果为空，则该名称可能可用。建议使用有描述性且独特的名称，可以考虑添加前缀或后缀使其更加独特。

### 如何处理包的版本号？

遵循[语义化版本规范](https://semver.org/lang/zh-CN/)：
- 主版本号（Major）：当你做了不兼容的API修改
- 次版本号（Minor）：当你做了向下兼容的功能性新增
- 修订号（Patch）：当你做了向下兼容的问题修正

例如：从1.2.3到2.0.0表示有破坏性变更，从1.2.3到1.3.0表示新增功能，从1.2.3到1.2.4表示修复bug。

### 如何在本地测试我的包安装？

您可以使用pip的开发模式安装：

```bash
pip install -e .
```

或者创建一个测试环境：

```bash
# 创建一个新的虚拟环境
python -m venv test_env
source test_env/bin/activate  # 在Windows上使用: test_env\Scripts\activate

# 从本地安装包
pip install /path/to/your/package

# 测试导入
python -c "import your_package_name; print(your_package_name.__version__)"
```

### 发布到PyPI时遇到权限问题怎么办？

确保您已经在PyPI上注册了账号，并且在`~/.pypirc`文件中配置了正确的凭证：

```
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = your_username
password = your_password

[testpypi]
repository = https://test.pypi.org/legacy/
username = your_username
password = your_password
```

或者使用环境变量：

```bash
export TWINE_USERNAME=your_username
export TWINE_PASSWORD=your_password
```

### 如何为我的包创建详细的文档？

1. 使用Sphinx生成文档：
   ```bash
   # 安装Sphinx
   pip install sphinx sphinx-rtd-theme
   
   # 在docs目录初始化Sphinx
   mkdir docs
   cd docs
   sphinx-quickstart
   ```

2. 配置`docs/conf.py`以自动生成API文档
3. 编写详细的模块、类和函数文档字符串
4. 使用Read the Docs或GitHub Pages托管生成的文档

### 如何处理不同Python版本的兼容性？

1. 在`pyproject.toml`中指定支持的Python版本：
   ```toml
   [project]
   requires-python = ">=3.7"
   ```

2. 使用条件导入处理版本差异：
   ```python
   import sys
   if sys.version_info >= (3, 8):
       from importlib import metadata
   else:
       import importlib_metadata as metadata
   ```

3. 使用tox测试多个Python版本：
   ```bash
   tox -e py37,py38,py39,py310,py311
   ```

### 我的包需要包含非Python文件（如数据文件）怎么办？

在`pyproject.toml`中配置包含的数据文件：

```toml
[tool.setuptools.package-data]
"your_package_name" = ["*.json", "data/*.csv", "templates/*.html"]
```

然后在代码中使用相对路径访问这些文件：

```python
import os
import pkg_resources

# 方法1：使用pkg_resources
data_path = pkg_resources.resource_filename('your_package_name', 'data/example.csv')

# 方法2：使用相对路径
data_path = os.path.join(os.path.dirname(__file__), 'data', 'example.csv')
```

# 添加可选依赖
[project.optional-dependencies]
vis = [
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
]
```

然后用户可以通过 `pip install your_package[vis]` 安装可选依赖。

### 如何运行特定的测试？

使用`pytest`的模式匹配功能：

```bash
# 运行特定测试文件
pytest tests/test_file.py

# 运行特定测试函数
pytest tests/test_file.py::test_function

# 运行标记的测试
pytest -m "slow"
```

### 如何生成API文档？

安装文档依赖并使用Sphinx：
```bash
# 安装文档依赖
pip install -e ".[docs]"

# 如果docs目录不存在，初始化它
mkdir -p docs
cd docs
sphinx-quickstart  # 按照提示配置

# 生成文档
make html
```

### 如何处理版本兼容性问题？

1. 在`pyproject.toml`中明确指定支持的Python版本
2. 使用条件导入处理不同版本的API差异：
   ```python
   import sys
   if sys.version_info >= (3, 10):
       from importlib.metadata import version
   else:
       from importlib_metadata import version
   ```
3. 使用`tox`测试不同Python版本的兼容性

### 如何添加新的CI/CD工作流？

1. 在`.github/workflows/`目录下创建新的YAML文件
2. 配置触发条件、运行环境和执行步骤
3. 推送到GitHub仓库，自动激活工作流

示例工作流（文档部署）：
```yaml
name: Deploy Docs

on:
  push:
    branches: [main]
    paths: ['docs/**']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -e ".[docs]"
      - name: Build docs
        run: |
          cd docs
          make html
      # 部署步骤...
```

### 如何优化包的性能？

1. 使用性能分析工具识别瓶颈：
   ```bash
   python -m cProfile -o profile.stats your_script.py
   python -m pstats profile.stats
   ```

2. 考虑使用Cython或Numba加速计算密集型代码：
   ```toml
   # 在pyproject.toml的optional-dependencies中
   perf = [
       "cython>=0.29",
       "numba>=0.53",
   ]
   ```

3. 实现并行处理：
   ```python
   from concurrent.futures import ProcessPoolExecutor
   
   def parallel_process(data_chunks):
       with ProcessPoolExecutor() as executor:
           return list(executor.map(process_function, data_chunks))
   ```

### 如何确保代码质量？

1. 使用pre-commit钩子自动运行代码检查
2. 设置测试覆盖率目标并监控
3. 进行定期代码审查
4. 使用静态类型检查（mypy）
5. 遵循PEP 8风格指南

### 如何处理包的依赖冲突？

1. 指定合适的版本范围，避免过于严格的版本限制
2. 使用虚拟环境隔离不同项目的依赖
3. 考虑使用Poetry或Conda等工具管理依赖
4. 在文档中明确说明已知的依赖冲突和解决方案