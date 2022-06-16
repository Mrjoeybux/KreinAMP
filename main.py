import argparse
import sys
import multiprocessing as mp
import comparison
import supplementary

if __name__ == '__main__':
	parser = argparse.ArgumentParser(add_help=False)
	subparsers = parser.add_subparsers(dest="command")

	predict_parser = subparsers.add_parser('test_set_pred')
	predict_parser.add_argument('-d', '--dataset', required=True)

	plot_parser = subparsers.add_parser('plot_lengths')

	dbaasp_parser = subparsers.add_parser('dbaasp_pred')

	model_compare = subparsers.add_parser('comparison')
	model_compare.add_argument('-d', '--dataset', required=True)
	model_compare.add_argument('-pr', '--print_log', nargs='?', default=0, required=False, type=int)
	model_compare.add_argument('-l', '--log_level', nargs='?', default="INFO", required=False, type=str)
	model_compare.add_argument('-s', '--score', nargs='?', default="default", required=False, type=str)
	model_compare.add_argument('-o', '--num_outer', nargs='?', default=10, required=False, type=int)
	model_compare.add_argument('-i', '--num_inner', nargs='?', default=10, required=False, type=int)
	model_compare.add_argument('-n', '--name', nargs='?', default=None, required=False, type=str)
	model_compare.add_argument('-hp', '--HPC', nargs='?', default=0, required=False, type=int)
	model_compare.add_argument('-mem', '--memory', required=False, default=5, type=int)
	model_compare.add_argument('-t', '--time', required=False, default=24, type=int)
	model_compare.add_argument('-c', '--num_cpus', required=False, default=mp.cpu_count(), type=int)
	model_compare.add_argument('-q', '--partition', required=False, default='defq', type=str)

	args = parser.parse_args(sys.argv[1:])

	if args.command == "test_set_pred":
		supplementary.predict(args)
	elif args.command == "plot_lengths":
		supplementary.plot_lengths()
	elif args.command == "dbaasp_pred":
		supplementary.dbaasp_predictions()
	elif args.command == "comparison":
		comparison.main(args)
