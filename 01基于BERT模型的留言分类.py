# fmt: off
import logging
from pathlib import Path

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

def doc_classification_cola():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    ml_logger = MLFlowLogger(tracking_uri="")
    ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_cola")

    ##########################
    ########## Settings
    ##########################
    set_all_seeds(seed=42)
    device, n_gpu = initialize_device_settings(use_cuda=False)
    n_epochs = 3
    batch_size = 8
    evaluate_every = 450
    lang_model = "/bert-base-chinese" #BERT中文模型的路径
    #模型下载地址https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
    do_lower_case = False

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=lang_model, do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    # Here we load Cola 2018 Data.

    label_list =["城乡建设","卫生计生","商贸旅游","劳动和社会保障","教育文体","交通运输","环境保护"]
    metric = "acc"

    processor = TextClassificationProcessor(tokenizer=tokenizer,
                                            max_seq_len=507,
                                            data_dir=Path("/BERT留言分类数据集"), #存放文本分类数据的文件夹路径，数据格式：第一列按字符分隔的text,第二列label，之间用制表符分隔。第一行需要有"text"与"label"
                                            dev_filename=None, #Path("dev.tsv"),
                                            dev_split=0.1,
                                            test_filename="/BERT留言分类数据集/test.tsv",
                                            label_list=label_list,
                                            metric=metric,
                                            label_column_name="label"
                                            )

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
    data_silo = DataSilo(
        processor=processor,
        batch_size=batch_size)

    # 4. Create an AdaptiveModel
    # a) which consists of a pretrained language model as a basis
    language_model = LanguageModel.load(lang_model)

    # language_model = Roberta.load(lang_model)
    # b) and a prediction head on top that is suited for our task => Text classification
    prediction_head = TextClassificationHead(
        num_labels=len(label_list),
        class_weights=data_silo.calculate_class_weights(task_name="text_classification"))

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[prediction_head],
        embeds_dropout_prob=0.1,
        lm_output_types=["per_sequence"],
        device=device)

    # 5. Create an optimizer
    model, optimizer, lr_schedule = initialize_optimizer(
        model=model,
        learning_rate=2e-5,
        device=device,
        n_batches=len(data_silo.loaders["train"]),
        n_epochs=n_epochs)

    # 6. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it from time to time
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=data_silo,
        epochs=n_epochs,
        n_gpu=n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=evaluate_every,
        device=device)

    # 7. Let it grow
    trainer.train()

    # 8. Hooray! You have a model. Store it:
    save_dir = Path("/BERT文本分类输出的模型")
    model.save(save_dir)
    processor.save(save_dir)


if __name__ == "__main__":
    doc_classification_cola()
