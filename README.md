# Code Explainer

This project provides a code explanation tool using OpenAI's language model to assist in understanding source code. It loads code from a specified repository, processes it, and allows users to query the codebase interactively.

## Requirements
Ensure you have the following installed:
- The required Python libraries listed in `requirements.txt`

## Setup

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/atikul-islam-sajib/codeExplainer.git
   cd codeExplainer
   ```

2. **Create a Virtual Environment**:
   ```sh
   python -m venv langchain-env
   source langchain-env/bin/activate
   ```

3. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the root directory of the project and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

5. **Configure the Project**:
   Update the `config.yml` file with the appropriate paths and settings. An example `config.yml`:
   ```yaml
    path:
        CODE_PATH: "./source_code/"
        DATA_PATH: "./data/"

    OpenAI:
        temperature: 1.0
        model_name: "gpt-3.5-turbo"

    vectorstores:
        Chroma: False
        FAISS: True

    sourcecode:
        url: https://github.com/your-repo-url

    analysis:
        folder: "src"
        filenames: "py"  # In this project I just used python

    chunks:
        chunk_size: 1000
        chunk_overlap: 200

    chatExplainer:
        chat_limit: 5
   ```

## Usage

### Running the Code Explainer

1. **Run DVC to Ensure Dependencies**:
   ```sh
   dvc repro
   ```

2. **Run the Explainer**:
   ```sh
   python src/codeExplainer.py --config ./config.yml --chat
   ```

3. **Interact with the Chat Explainer**:
   - When prompted, input your queries to understand the source code.
   - The chat limit is defined in the `config.yml` file.

### Example Commands

```sh
python src/codeExplainer.py --config ./config.yml --chat
```

### Directory Structure

```
codeExplainer/
│
├── src/
│   ├── codeExplainer.py
│   ├── utils.py
│   ├── template.py
│
├── .env
├── config.yml
├── dvc.yaml
├── requirements.txt
├── README.md
```

## Troubleshooting

- **ModuleNotFoundError**: Ensure all dependencies are installed in the correct Python environment.
- **DVC Errors**: Check the `dvc.yaml` file for correct formatting and paths.
- **Configuration Errors**: Ensure all paths and settings in `config.yml` are correctly specified.

## Contributing

If you want to contribute to this project, please fork the repository and submit pull requests. Ensure your code adheres to the project's coding standards.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.