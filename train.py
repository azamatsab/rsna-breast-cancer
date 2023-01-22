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
        default="outputs/tf_efficientnetv2_l/1_1/weights/tf_efficientnetv2_l_8_0.7632_0.3733.pth",
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

    for i in range(0, 5):
        config = get_configs(args.config_path)

        # Ugly trick to dump config into res dir to preserve hyperparameters
        config["self_path"] = args.config_path
        config["cam"] = args.cam
        # config["fold"] = i
        arch = getattr(models, config["model"])
        model = arch(config)

        train_tr = model.train_transform()
        test_tr = model.test_transform()
        train_loader, test_loader = get_loaders(config, train_tr, test_tr)
        model.set_scheduler(len(train_loader))

        if args.cam:
            config["batch_size"] = 4
            _, test_loader = get_loaders(config, train_tr, test_tr)
            model.load(args.weights_path)
            cam = CamCalculator(config, model)
            cam.calculate_cam(test_loader)
        elif args.eval:
            model.load(args.weights_path)
            trainer = Trainer(model, config)
            trainer.evaluate(test_loader)
        else:
            if args.resume:
                model.load(args.weights_path)

            trainer = Trainer(model, config)
            trainer.run(train_loader, test_loader)
        
        break
