#!/usr/bin/env python3
"""
Deep Learning Tutorial - CLI 交互式入口

提供命令行界面，方便用户浏览和运行教学模块。
"""

import sys
import importlib
import glob
import os


def discover_modules():
    """
    动态发现当前目录下所有可执行的教学模块

    Returns:
        list: 包含 (module_name, module_path, description) 的元组列表
    """
    modules = []

    # 查找所有 deep_learning_*.py 文件
    pattern = "deep_learning_*.py"
    files = sorted(glob.glob(pattern))

    for file_path in files:
        module_name = os.path.splitext(file_path)[0]

        # 跳过练习文件，它们有单独的处理
        if "exercises" in module_name:
            continue

        # 尝试导入模块获取描述
        try:
            module = importlib.import_module(module_name)
            # 安全处理 __doc__，避免 None.strip() 异常
            doc = module.__doc__ if hasattr(module, '__doc__') and module.__doc__ else ''
            description = doc.strip().split('\n')[0] if doc.strip() else "无描述"
        except Exception as e:
            # 输出真实异常以便调试
            description = f"模块加载失败: {type(e).__name__}"

        modules.append((module_name, file_path, description))

    # 添加练习模块
    if os.path.exists("deep_learning_exercises.py"):
        modules.append(("deep_learning_exercises", "deep_learning_exercises.py", "深度学习练习题"))

    return modules


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        Deep Learning Tutorial - 深度学习教程              ║
║                                                           ║
║        纯 Python 实现，无需深度学习框架                   ║
║        帮助理解深度学习底层原理                           ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_menu(modules):
    """
    打印教学模块菜单

    Args:
        modules: 模块列表
    """
    print("\n可用的教学模块:\n")

    for idx, (module_name, _, description) in enumerate(modules, 1):
        # 格式化模块名（移除前缀，使用更友好的显示）
        display_name = module_name.replace("deep_learning_", "").replace("_", " ").title()
        print(f"  {idx}. {display_name:30s} - {description}")

    print(f"\n  0. 退出程序")
    print("-" * 65)


def run_module(module_name):
    """
    运行指定的教学模块

    Args:
        module_name: 模块名称
    """
    try:
        print(f"\n{'='*65}")
        print(f"正在运行: {module_name}")
        print(f"{'='*65}\n")

        module = importlib.import_module(module_name)

        # 查找并运行 main() 函数
        if hasattr(module, 'main'):
            module.main()
        # 如果有 __main__ 代码块，重新执行模块
        elif hasattr(module, '__name__'):
            exec(open(f"{module_name.replace('.', '/')}.py").read())
        else:
            print(f"警告: 模块 {module_name} 没有 main() 函数")
            print("提示: 你可以直接导入该模块使用其中的类和函数")

        print(f"\n{'='*65}")
        print(f"模块 {module_name} 运行完成")
        print(f"{'='*65}\n")

    except Exception as e:
        print(f"\n错误: 运行模块 {module_name} 时出错")
        print(f"错误信息: {e}")
        print("\n提示: 确保已安装所有依赖 (pip install -r requirements.txt)")


def show_module_info(modules):
    """
    显示所有模块的详细信息

    Args:
        modules: 模块列表
    """
    print("\n" + "="*65)
    print("教学模块详细信息")
    print("="*65 + "\n")

    for module_name, file_path, description in modules:
        print(f"模块名称: {module_name}")
        print(f"文件路径: {file_path}")
        print(f"描述信息: {description}")

        # 尝试获取更多信息
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, '__doc__') and module.__doc__:
                doc = module.__doc__.strip()
                if len(doc) > 100:
                    doc = doc[:100] + "..."
                print(f"文档说明: {doc}")
        except Exception:
            pass

        print("-" * 65)


def interactive_mode():
    """交互式模式主循环"""
    print_banner()

    # 发现所有可用模块
    modules = discover_modules()

    if not modules:
        print("错误: 未找到任何教学模块")
        print("请确保当前目录包含 deep_learning_*.py 文件")
        return

    while True:
        print_menu(modules)

        try:
            choice = input("\n请选择模块编号 (或输入 'help' 查看详细信息): ").strip()

            if choice.lower() in ['q', 'quit', 'exit', '0']:
                print("\n感谢使用 Deep Learning Tutorial!")
                print("继续学习，探索深度学习的奥秘！\n")
                break

            elif choice.lower() == 'help':
                show_module_info(modules)
                continue

            elif choice.lower() in ['list', 'ls']:
                # 仅显示菜单，继续循环
                continue

            try:
                idx = int(choice)
                if 1 <= idx <= len(modules):
                    module_name = modules[idx - 1][0]
                    run_module(module_name)

                    # 询问是否继续
                    response = input("\n按 Enter 键返回菜单 (或输入 'q' 退出): ").strip()
                    if response.lower() in ['q', 'quit', 'exit']:
                        print("\n感谢使用 Deep Learning Tutorial!\n")
                        break
                else:
                    print(f"\n错误: 请输入 0-{len(modules)} 之间的数字")

            except ValueError:
                print("\n错误: 无效的输入，请输入数字或命令")

        except KeyboardInterrupt:
            print("\n\n检测到中断信号，退出程序...")
            print("感谢使用 Deep Learning Tutorial!\n")
            break

        except Exception as e:
            print(f"\n错误: {e}")
            print("按 Enter 键继续...")
            input()


def cli_mode(args):
    """
    命令行模式

    Args:
        args: 命令行参数列表
    """
    if len(args) == 0 or args[0] in ['-h', '--help']:
        print("用法: python main.py [选项] [模块名]")
        print("\n选项:")
        print("  -h, --help     显示此帮助信息")
        print("  -l, --list     列出所有可用模块")
        print("  -i, --interactive  进入交互模式 (默认)")
        print("\n示例:")
        print("  python main.py                    # 交互模式")
        print("  python main.py --list             # 列出所有模块")
        print("  python main.py fundamentals       # 运行基础模块")
        return

    if args[0] in ['-l', '--list']:
        modules = discover_modules()
        print("\n可用的教学模块:\n")
        for module_name, file_path, description in modules:
            print(f"  {module_name:40s} - {description}")
        print()
        return

    # 尝试运行指定模块
    module_name = args[0]
    if not module_name.startswith('deep_learning_'):
        module_name = f'deep_learning_{module_name}'

    run_module(module_name)


def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        cli_mode(sys.argv[1:])
    else:
        # 默认进入交互模式
        interactive_mode()


if __name__ == "__main__":
    main()
