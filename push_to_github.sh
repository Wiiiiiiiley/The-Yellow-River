#!/bin/bash
# 推送到 GitHub 并创建新分支的脚本

echo "======================================"
echo "推送到 GitHub 并创建新分支"
echo "======================================"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在 git 仓库中
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "错误: 当前目录不是 git 仓库"
    exit 1
fi

# 获取当前分支
current_branch=$(git branch --show-current)
echo -e "${YELLOW}当前分支: $current_branch${NC}"

# 询问新分支名称
echo ""
read -p "请输入新分支名称 (例如: feature/enhanced-pinn-v3): " new_branch

if [ -z "$new_branch" ]; then
    echo "错误: 分支名称不能为空"
    exit 1
fi

echo ""
echo "步骤 1: 创建并切换到新分支 '$new_branch'..."
git checkout -b "$new_branch"
if [ $? -ne 0 ]; then
    echo "错误: 创建分支失败"
    exit 1
fi

echo ""
echo "步骤 2: 添加所有更改到暂存区..."
git add -A

echo ""
echo "步骤 3: 检查更改状态..."
git status --short

echo ""
read -p "请输入提交信息 (默认: 'Add enhanced PINN v3.0 with all improvements'): " commit_msg
commit_msg=${commit_msg:-"Add enhanced PINN v3.0 with all improvements"}

echo ""
echo "步骤 4: 提交更改..."
git commit -m "$commit_msg"
if [ $? -ne 0 ]; then
    echo "警告: 提交失败或没有更改要提交"
fi

echo ""
echo "步骤 5: 检查远程仓库..."
git remote -v

echo ""
read -p "是否推送到 GitHub? (y/n): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo ""
    echo "步骤 6: 推送到 GitHub..."
    git push -u origin "$new_branch"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}======================================"
        echo -e "✅ 推送成功!"
        echo -e "======================================${NC}"
        echo "分支: $new_branch"
        echo "提交信息: $commit_msg"
        echo ""
        echo "GitHub 链接:"
        # 尝试提取远程 URL 并生成链接
        remote_url=$(git remote get-url origin 2>/dev/null)
        if [ ! -z "$remote_url" ]; then
            # 转换 SSH URL 为 HTTPS URL
            if [[ $remote_url == git@github.com:* ]]; then
                repo_path=${remote_url#git@github.com:}
                repo_path=${repo_path%.git}
                echo "https://github.com/$repo_path/tree/$new_branch"
            elif [[ $remote_url == https://github.com/* ]]; then
                repo_path=${remote_url%.git}
                echo "$repo_path/tree/$new_branch"
            fi
        fi
    else
        echo ""
        echo "❌ 推送失败"
        echo "请检查:"
        echo "1. 是否已配置 GitHub 远程仓库 (git remote add origin <url>)"
        echo "2. 是否有推送权限"
        echo "3. 网络连接是否正常"
    fi
else
    echo ""
    echo "已取消推送"
    echo "本地分支 '$new_branch' 已创建并提交"
fi

echo ""
echo "======================================"
echo "当前分支: $(git branch --show-current)"
echo "======================================"
