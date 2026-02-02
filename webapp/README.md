# 智能合约检测 Web App

## 运行方式

1. 安装依赖（若尚未安装 Flask）：
   ```bash
   pip install flask
   ```

2. 在 **FYP 项目根目录** 下启动（不要只在 webapp 目录下）：
   ```bash
   cd C:\Users\john\Desktop\FYP
   python webapp/app.py
   ```

3. 浏览器打开：**http://127.0.0.1:5000**

4. 在输入框输入智能合约地址（如 `0xdC94E8Ab22d66bcC9b0BDB5E48711Fb12CBea74e`），点击「开始检测」。  
   检测在后台执行（约 1–2 分钟），页面会轮询并自动显示结果。

## 说明

- 后端会依次执行：下载 Bytecode → Vandal 反编译 → GNN 检测。
- 检测完成后会显示：恶意评分、模型信心、结论（高危 / 疑似异常 / 安全）。
