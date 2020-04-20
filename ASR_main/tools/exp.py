
class Experiment():
    '''
    ------------------
    Data descriptors:
    dirs
        Hold all directory information to be used in the experiment
        in dict() form.
        To see all directories that are related, refer to:
        list(self.dirs.items())
    model_p
        Hold all parameters related to NN model learning in dict() form
    train_p
        Hold all parameters related to training in dict() form
    speaker_list
        Hold all name of speakers
    '''
    def __init__(self, num_speakers = 100, exp_name = None, model_p = None, new = True):
        # 1] Initialize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        np.random.seed(0)
        torch.manual_seed(0)
        self.create_env(exp_name, new)
        # self.speaker_list = sorted(os.listdir(self.dirs['train_data']))
        self.speaker_list = ['p225','p226','p227','p228']
        self.num_speakers = len(self.speaker_list)
        assert self.num_speakers == num_speakers, 'Specified "num_speakers" and "num_speakers in train data" does not match'
        self.build_model(params = model_p)
        self.model_p = model_p
        self.p = Printer(filewrite_dir = self.dirs['log'])

        # 2] Hyperparameters for Training
        self.train_p = dict()
        self.train_p['n_train_frames'] = 128
        self.train_p['iter_per_ep'] = 4
        self.train_p['start_epoch'] = 1
        self.train_p['n_epoch'] = 100
        self.train_p['batch_size'] = 8
        self.train_p['model_save_epoch'] = 1
        self.train_p['validation_epoch'] = 1
        self.lambd = dict(
        SI = 1,
        LI = 1,
        AC = 1,
        SC = 1,
        C = 1,
        )
        self.preprocess_p = dict(
        sr = 16000,
        frame_period = 5.0,
        num_mcep = 36,
        )

        # 3] Hyperparameters for saving model
        self.model_kept = []
        self.max_keep=100

        # 4] If the experiment is not new, Load most recent model
        if new == False:
            self.model_kept= sorted(os.listdir(self.dirs['model']))
            most_trained_model = max(model_list)
            epoch_trained = int(most_trained_model.split('-')[-1])
            self.train_p['start_epoch'] += epoch_trained
            print('Loading model from %s'%most_trained_model)
            self.load_model_all(os.path.join(self.dirs['model'], most_trained_model))

    def create_env(self, exp_name = None, new = True):
        '''Create experiment environment
        Store all "static directories" required for experiment in "self.dirs"(dict)

        Store every experiment result in: exp/exp_name/ == exp_dir
        including log, model, test(validation) etc
        '''
        # 0] exp_dir == master directory
        self.dirs = dict()
        exp_dir = 'exp/'
        model_dir = 'model/'
        log_dir = 'log.txt'
        train_data_dir = 'processed/'
        si_dir = 'processed_stateindex/'
        # Validation
        validation_data_dir = 'processed_validation/inset_dev/'
        validation_pathlist_dir = 'filelist/inset_dev.lst'
        validation_dir = 'validation/'
        validation_result_dir = 'validation_result/'
        validation_log_dir = 'validation_log.txt'
        # Test
        test_dir = 'test/'
        test_result_dir = 'test_result/'
        test_converted_dir = 'converted/'
        test_data_dir = 'processed_validation/inset_test/'
        test_pathlist_dir = 'filelist/inset_test.lst'

        # 1] Set up Experiment directory
        if exp_name == None:
            exp_name = time.strftime('%m%d_%H%M%S')
        self.dirs['exp'] = os.path.join(exp_dir, exp_name)
        if new == True:
            assert not os.path.isdir(self.dirs['exp']), 'New experiment, but exp_dir with same name exists'
            os.makedirs(self.dirs['exp'])
        else:
            assert os.path.isdir(self.dirs['exp']), 'Existing experiment, but exp_dir doesn\'t exist'

        # 2] Model parameter directory
        self.dirs['model'] = os.path.join(self.dirs['exp'], model_dir)
        os.makedirs(self.dirs['model'], exist_ok=True)

        # 3] Log settings
        self.dirs['log'] = os.path.join(self.dirs['exp'], log_dir)

        # 4] Train data
        self.dirs['train_data'] = train_data_dir
        self.dirs['si'] = si_dir

        # 5] Test (Including Validation)
        self.dirs['validation_data'] = validation_data_dir
        self.dirs['validation_pathlist'] = validation_pathlist_dir

        # 6] Validation
        self.dirs['validation'] = os.path.join(self.dirs['exp'], validation_dir)
        self.dirs['validation_result'] = os.path.join(self.dirs['validation'], validation_result_dir)
        self.dirs['validation_log'] = os.path.join(self.dirs['validation'], validation_log_dir)
        os.makedirs(self.dirs['validation'], exist_ok=True)
        os.makedirs(self.dirs['validation_result'], exist_ok=True)

        # 7] Test
        self.dirs['test'] = os.path.join(self.dirs['exp'], test_dir)
        self.dirs['test_result'] = os.path.join(self.dirs['test'], test_result_dir)
        self.dirs['test_converted'] = os.path.join(self.dirs['test'], test_converted_dir)
        self.dirs['test_data'] = os.path.join(self.dirs['test'], test_data_dir)
        self.dirs['test_pathlist'] = os.path.join(self.dirs['test'], test_pathlist_dir)
        os.makedirs(self.dirs['test'], exist_ok=True)
        os.makedirs(self.dirs['test_result'], exist_ok=True)
        os.makedirs(self.dirs['test_converted'], exist_ok=True)
