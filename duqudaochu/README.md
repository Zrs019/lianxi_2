# 冷热量表历史记录分批导出工具

这个项目放在 `D:\minicondadaima\lianxi\duqudaochu`，用于自动操作“和达能源仪表平台”的“冷热量表历史记录”页面：

- 运行后由你选择设备目录、开始时间、结束时间、查询间隔（时/日/月/年）。
- 设备选择限制在二级目录，例如 `中心站-维亚园区`。
- 网站单次查询最多 100 台设备，脚本会自动拆成多批导出。
- 多个批次会自动合并成一个文件。
- 最终文件名格式为：`设备_起始时间_结束时间.xlsx`。

## 安装依赖

在 PowerShell 中运行：

```powershell
cd D:\minicondadaima\lianxi\duqudaochu
.\install.ps1
```

如果 PowerShell 弹出“选择打开方式”，改用批处理：

```powershell
cd D:\minicondadaima\lianxi\duqudaochu
.\install.bat
```

如果你的 `python` 没有加入 PATH，可以传入 Python 完整路径：

```powershell
.\install.ps1 -Python "D:\你的Python路径\python.exe"
```

## 配置网站地址

打开 `config.yaml`，把 `website_url` 改成平台地址，例如：

```yaml
website_url: "http://你的平台地址"
```

如果你不填，脚本启动后也会在命令行里问你。

## 运行

```powershell
cd D:\minicondadaima\lianxi\duqudaochu
.\run.ps1
```

如果 PowerShell 弹出“选择打开方式”，改用：

```powershell
cd D:\minicondadaima\lianxi\duqudaochu
.\run.bat
```

如果需要指定 Python：

```powershell
.\run.ps1 -Python "D:\你的Python路径\python.exe"
```

批处理也支持指定 Python：

```powershell
.\run.bat "D:\你的Python路径\python.exe"
```

脚本会打开浏览器。第一次使用时请手动登录并进入“冷热量表历史记录”页面，然后回到命令行按回车。之后脚本会读取左侧设备树，让你选择二级目录。

## 接口导出

现在默认 `export_mode: "api"`，也就是用后端接口下载，不再逐个模拟点击导出。

第一次使用接口导出前，需要抓一次真实导出接口：

```powershell
cd D:\minicondadaima\lianxi\duqudaochu
.\capture_api.bat
```

浏览器打开后，手动选择一个小目录，设置时间和间隔，点击导出并选择“不含图表”。下载开始或完成后，回到命令行按回车。脚本会生成 `api_capture.yaml`。

然后正常运行：

```powershell
.\run.bat
```

每次成功导出后，命令行会询问是否继续。输入 `y` 继续下一次导出；直接回车或输入 `n` 结束程序。

如果接口参数名自动识别失败，按命令行提示把 `api_capture.yaml` 里的参数名填到 `config.yaml`：

```yaml
api:
  parameter_keys:
    device_ids: "真实设备参数名"
    start_time: "真实开始时间参数名"
    end_time: "真实结束时间参数名"
    interval: "真实查询间隔参数名"
    export_type: "真实导出类型参数名"
```

设备分批策略：

- 不超过 100 台时，优先直接用二级目录导出。
- 没有二级目录时，用三级目录。
- 超过 100 台时，才拆成单台设备分批导出。

默认会用 Microsoft Edge 打开。对应配置在 `config.yaml`：

```yaml
browser_channel: "msedge"
```

如果想改回 Playwright 自带 Chromium，把它改成空字符串：

```yaml
browser_channel: ""
```

时间输入格式建议使用页面一致的格式：

```text
2026-04-28 16:27:02
```

查询间隔可输入：

```text
时 / 日 / 月 / 年
```

也可以输入：

```text
hour / day / month / year
```

## 输出目录

- 分批下载文件：`downloads\batches`
- 合并后的文件：`output`

如果 `keep_batch_files: false`，合并后会删除分批下载文件。

## 调试

如果脚本提示找不到设备树、时间输入框或导出按钮，通常是页面 CSS 和默认配置不完全一致。先运行：

```powershell
.\run.ps1 -Inspect
```

它会在你登录并打开页面后打印页面中可能相关的控件文本，方便修改 `config.yaml`。

常见需要调整的位置：

- `selectors.tree`
- `selectors.time_input_selector`
- `selectors.export_dropdown_trigger`
- `buttons.query`
- `buttons.export`

## 注意

脚本会保存浏览器登录状态到 `browser_profile`，不要把这个目录发给别人。
