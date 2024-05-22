---
# Medical Chatbot

This project is a medical chatbot developed using Python, Chainer, Lanchain, Transformers, and the Llama2 model. The chatbot is designed to provide users with personalized medical recommendations, information about symptoms, cures, precautions, and first aid medicines.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)

## Installation

To use the Medical Chatbot, you'll need to have Python and the following libraries installed:

- Chainlit
- Langchain
- Transformers
- Llama2

You can install these using pip:

```bash
pip install chainlit langchain ctransformers langchain_community
```

## Usage

To start the chatbot, simply run the main Python script:

```bash
chainlit run model.py
```

You can then interact with the chatbot by typing in your queries or symptoms.

## Training

The chatbot is trained using PDF files. To train the chatbot with your own data, place your PDF files in the `data` directory and run the training script:

```bash
python ingest.py
```

## Contributing

We welcome contributions from the community. If you'd like to contribute, please fork the repository and make changes as you'd like. Pull requests are warmly welcome.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

---
