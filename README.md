# AgentMinusMinus

Agent controllers for reducing repeated language-model reasoning through supported decisions and guarded native execution.

The complete CHISEL Python package, integration documentation, source mappings, synthetic tests, and publication checks are in [CHISEL](CHISEL/README.md).

## Quick start

```bash
git clone git@github.com:microsoft/AgentMinusMinus.git
cd AgentMinusMinus/CHISEL
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python tools/check_release.py
python -m pytest -q
```

Install Node.js 22 or later for the optional native bank bridge and its tests. Tests use synthetic inputs and mocked responses.

## License

The repository template retains its MIT [license](LICENSE). The CHISEL component is distributed under [Apache-2.0](CHISEL/LICENSE), with its original notices and source mappings retained.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
