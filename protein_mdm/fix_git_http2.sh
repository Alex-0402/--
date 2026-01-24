#!/bin/bash
# 修复 Git HTTP/2 错误的脚本
# 解决 "RPC failed; curl 16 Error in the HTTP2 framing layer" 错误

echo "=========================================="
echo "修复 Git HTTP/2 协议错误"
echo "=========================================="

# 方法 1: 设置全局配置（推荐）
echo ""
echo "正在设置全局 Git 配置..."
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000

if [ $? -eq 0 ]; then
    echo "✅ 全局配置设置成功"
else
    echo "⚠️  全局配置设置失败，尝试仅设置当前仓库配置..."
    git config http.version HTTP/1.1
    git config http.postBuffer 524288000
    if [ $? -eq 0 ]; then
        echo "✅ 当前仓库配置设置成功"
    else
        echo "❌ 配置设置失败，请手动执行："
        echo "   git config --global http.version HTTP/1.1"
        echo "   git config --global http.postBuffer 524288000"
    fi
fi

echo ""
echo "当前 HTTP 配置："
git config --get http.version || echo "  http.version: 未设置（将使用默认值）"
git config --get http.postBuffer || echo "  http.postBuffer: 未设置"

echo ""
echo "=========================================="
echo "现在可以尝试执行："
echo "  git pull --tags origin main"
echo ""
echo "如果仍然失败，可以尝试："
echo "  1. 使用 SSH: git remote set-url origin git@github.com:Alex-0402/--.git"
echo "  2. 使用环境变量: GIT_HTTP_VERSION=1.1 git pull --tags origin main"
echo "=========================================="
