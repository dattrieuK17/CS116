def main():
#     parser = argparse.ArgumentParser()
#     subparsers = parser.add_subparsers(dest='command')

#     # Train command
#     parser_train = subparsers.add_parser('train')
#     parser_train.add_argument('--public_dir', type=str)
#     parser_train.add_argument('--model_dir', type=str)

#     # Predict command
#     parser_predict = subparsers.add_parser('predict')
#     parser_predict.add_argument('--model_dir', type=str)
#     parser_predict.add_argument('--test_input_dir', type=str)
#     parser_predict.add_argument('--output_path', type=str)

#     args = parser.parse_args()

#     if args.command == 'train':
#         run_train(args.public_dir, args.model_dir)
#     elif args.command == 'predict':
#         run_predict(args.model_dir, args.test_input_dir, args.output_path)
#     else:
#         parser.print_help()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()