# Python PyPI 库模板

这是一个用于创建可发布到PyPI的Python库的模板项目。它提供了一个完整的项目结构和配置，使您能够快速开始开发自己的Python库。

本模板旨在解决Python库开发中的常见问题，提供标准化的项目结构和最佳实践，帮助开发者专注于核心功能的实现，而不是项目配置和结构设计。无论您是开发工具库、数据处理包还是Web框架，本模板都能为您提供坚实的基础。

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

将此模板转换为您自己的项目需要几个关键步骤。以下是一个完整的转换流程：

### 第1步：项目初始化

1. 确保您已经使用模板创建了新仓库并克隆到本地
2. 创建并激活虚拟环境
   ```bash
   python -m venv venv
   source venv/bin/activate  # 在Windows上使用: venv\Scripts\activate
   ```
3. 安装开发依赖
   ```bash
   pip install -e ".[dev]"
   ```
4. 初始化Git钩子（可选但推荐）
   ```bash
   pre-commit install
   ```

### 第2步：基础配置

1. 在`pyproject.toml`中修改项目元数据：
   - `name`: 您的包名（确保在PyPI上是唯一的）
   - `version`: 版本号（遵循[语义化版本规范](https://semver.org/lang/zh-CN/)）
   - `description`: 项目简短描述
   - `authors`: 作者信息
   - `dependencies`: 项目依赖项
   - `optional-dependencies`: 可选依赖项
   - `project.urls`: 更新项目相关链接
   
   示例：
   ```toml
   [project]
   name = "your_package_name"
   version = "0.1.0"
   description = "您的项目描述"
   authors = [
       {name = "您的姓名", email = "您的邮箱@example.com"}
   ]
   
   [project.urls]
   "Homepage" = "https://github.com/您的用户名/您的仓库名"
   "Bug Tracker" = "https://github.com/您的用户名/您的仓库名/issues"
   ```

2. 重命名包目录：
   ```bash
   mv src/example_package src/your_package_name
   ```
   
   然后更新导入语句和引用。可以使用以下命令在项目中查找所有需要更新的引用：
   ```bash
   grep -r "example_package" .
   ```

3. 更新包的初始化文件：
   - 修改`src/your_package_name/__init__.py`中的版本号
   - 添加需要导出的类和函数
   - 配置`__all__`列表
   
   示例：
   ```python
   """您的包描述。

   简要说明包的用途和功能。
   """

   __version__ = "0.1.0"
   
   from .core import ExampleClass, utility_function
   
   __all__ = ["ExampleClass", "utility_function"]
   ```

4. 更新许可证信息：
   - 修改`LICENSE`文件中的版权所有者和年份
   - 如需要，更换其他开源许可证（如MIT、GPL等）
   - 同时更新`pyproject.toml`中的license字段

5. 更新文档：
   - 修改本`README.md`文件，包括项目描述、安装说明和使用示例
   - 编写详细的API文档（可使用Sphinx自动生成）
   - 添加使用示例和教程
   - 创建CHANGELOG.md记录版本变更

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