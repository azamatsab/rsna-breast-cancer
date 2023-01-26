import warnings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging

# sys.path.insert(0, os.path.abspath(Path(__file__).parents[1].resolve()))
from src.trainer import Trainer
from src.utils import get_configs
import src.models as models
from src.datasets import get_loaders
from src.cam import CamCalculator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


def setup_parser(parser):
    parser.add_argument(
        "--configs",
        default="confs/config.yml",
        help="path to load configs",
        dest="config_path",
    )
    parser.add_argument("--eval", default=0, help="evaluation mode", dest="eval")
    parser.add_argument("--resume", default=0, help="continue training", dest="resume")
    parser.add_argument("--cam", default=0, help="use grad-cam on the val dataset", dest="cam")
    parser.add_argument(
        "--weights_path",
        default="outputs/tf_efficientnetv2_s/87_1/weights/swa.pth",
        help="model weights path",
        dest="weights_path",
    )
    return parser


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Classificator trainer",
        description="pipeline to train classification models",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser = setup_parser(parser)
    args = parser.parse_args()

    config = get_configs(args.config_path)

    # Ugly trick to dump config into res dir to preserve hyperparameters
    config["self_path"] = args.config_path
    config["cam"] = args.cam
    arch = getattr(models, config["model"])
    model = arch(config)
    train_tr = model.train_transform()
    test_tr = model.test_transform()

    if args.cam:
        config["batch_size"] = 4
        _, test_loader = get_loaders(config, train_tr, test_tr)
        model.load(args.weights_path)
        cam = CamCalculator(config, model)
        cam.calculate_cam(test_loader)
    elif args.eval:
        # paths = [
        #     "outputs/tf_efficientnetv2_s/43_0/weights/swa.pth",
        #     "outputs/tf_efficientnetv2_s/43_0/weights/tf_efficientnetv2_s_7_0.7637_0.441.pth",
        #     "outputs/tf_efficientnetv2_s/43_0/weights/tf_efficientnetv2_s_8_0.769_0.4343.pth",
        #     "outputs/tf_efficientnetv2_s/43_0/weights/tf_efficientnetv2_s_9_0.7623_0.4365.pth",
        #     "outputs/tf_efficientnetv2_s/43_0/weights/tf_efficientnetv2_s_10_0.766_0.441.pth",
        #     "outputs/tf_efficientnetv2_s/43_0/weights/tf_efficientnetv2_s_11_0.7739_0.4435.pth",
        #     "outputs/tf_efficientnetv2_s/43_0/weights/tf_efficientnetv2_s_12_0.7656_0.4411.pth",
        #     "outputs/tf_efficientnetv2_s/43_0/weights/tf_efficientnetv2_s_13_0.7714_0.4544.pth",
        #     "outputs/tf_efficientnetv2_s/43_0/weights/tf_efficientnetv2_s_14_0.7663_0.438.pth",
        #     "outputs/tf_efficientnetv2_s/43_0/weights/tf_efficientnetv2_s_15_0.774_0.4374.pth",
        #     "outputs/tf_efficientnetv2_s/43_0/weights/tf_efficientnetv2_s_16_0.7749_0.4411.pth"
        # ]
        # for weights_path in paths:
        #     print(weights_path)
        #     for site in [1, 2]:
        #         config["site_id"] = site
        #         _, test_loader = get_loaders(config, train_tr, test_tr)
        #         # model.load(args.weights_path)
        #         model.load(weights_path)
        #         trainer = Trainer(model, config)
        #         trainer.evaluate(test_loader)

        # config["tta"] = 1
        _, test_loader = get_loaders(config, train_tr, test_tr)
        model.load(args.weights_path)
        trainer = Trainer(model, config)
        trainer.evaluate(test_loader)

    else:
        train_loader, test_loader = get_loaders(config, train_tr, test_tr)
        model.set_scheduler(len(train_loader))
        if args.resume:
            model.load(args.weights_path)

        trainer = Trainer(model, config)
        trainer.run(train_loader, test_loader)
