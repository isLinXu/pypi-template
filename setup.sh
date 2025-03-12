#!/bin/bash

# 设置错误时退出和管道错误检测
set -e
set -o pipefail

# 颜色定义
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
NC="\033[0m" # No Color

# 打印带颜色的信息
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        error "未找到命令: $1"
    fi
}

# 验证包名格式
validate_package_name() {
    if [[ ! $1 =~ ^[a-z][a-z0-9_]*$ ]]; then
        error "包名无效: $1\n包名必须以小写字母开头，只能包含小写字母、数字和下划线"
    fi
}

# 检查必要的命令
check_command python3
check_command git

# 获取用户输入
while true; do
    read -p "请输入您的包名 (例如: my_package): " package_name
    if [ -z "$package_name" ]; then
        warn "包名不能为空，请重新输入"
        continue
    fi
    validate_package_name "$package_name" && break
done

while true; do
    read -p "请输入您的 GitHub 用户名: " github_username
    if [ -z "$github_username" ]; then
        warn "GitHub 用户名不能为空，请重新输入"
        continue
    fi
    break
done

read -p "请输入仓库名称 (默认与包名相同): " repo_name
repo_name=${repo_name:-$package_name}

read -p "请输入项目描述: " project_description
project_description=${project_description:-"A Python package for ${package_name}"}

read -p "请输入您的姓名: " author_name
read -p "请输入您的邮箱: " author_email

info "开始项目初始化..."

# 创建并激活虚拟环境
info "创建虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 重命名包目录
info "重命名包目录..."
if [ -d "src/example_package" ]; then
    mv src/example_package "src/${package_name}"
fi

# 更新文件中的包名引用
info "更新包名引用..."
find . -type f -name "*.py" -o -name "*.yml" -o -name "*.toml" | while read file; do
    sed -i '' "s/example_package/${package_name}/g" "$file"
done

# 安装开发依赖
info "安装开发依赖..."
pip install -e ".[dev]"

# 初始化 pre-commit 钩子
info "初始化 Git 钩子..."
pre-commit install

# 创建文档目录
info "创建文档目录..."
mkdir -p docs

# 更新 pyproject.toml
info "更新 pyproject.toml..."
sed -i '' \
    -e "s/name = \"example_package\"/name = \"${package_name}\"/" \
    -e "s/description = \".*\"/description = \"${project_description}\"/" \
    -e "s/Your Name/${author_name}/" \
    -e "s/your\.email@example\.com/${author_email}/" \
    -e "s|yourusername/example_package|${github_username}/${repo_name}|g" \
    pyproject.toml

# 创建 CHANGELOG.md
info "创建 CHANGELOG.md..."
cat > CHANGELOG.md << EOL
# 更新日志

## 0.1.0 ($(date +%Y-%m-%d))

- 初始版本发布
EOL

# 更新 README.md
info "更新 README.md..."
sed -i '' \
    -e "s/example_package/${package_name}/g" \
    -e "s/您的用户名/${github_username}/g" \
    -e "s/您的仓库名/${repo_name}/g" \
    README.md

info "\n项目初始化完成！\n"
info "后续步骤："
echo "1. 检查 pyproject.toml 中的配置是否正确"
echo "2. 编辑 src/${package_name}/__init__.py 更新包文档"
echo "3. 开始编写您的代码！"

warn "注意：您当前处于虚拟环境中。要退出虚拟环，请运行 'deactivate'"