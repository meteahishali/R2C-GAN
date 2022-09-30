import tensorflow as tf

import module
import utils

class r2c_gan:
    def __init__(self):
        self.G_A2B = None
        self.G_B2A = None

        self.D_A = None
        self.D_B = None

        self.d_loss_fn, self.g_loss_fn = utils.lsgan_loss()
        self.cycle_loss_fn = tf.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.losses.MeanAbsoluteError()
        self.class_loss_fn = tf.losses.CategoricalCrossentropy()

        self.G_lr_scheduler = None
        self.D_lr_scheduler = None
        self.G_optimizer = None
        self.D_optimizer = None
        
        self.cycle_weights = None
        self.identity_weight = None

        self.filter = None

    def init(self, args, len_dataset):

        self.filter = args['method']

        self.cycle_weights = args['cycle_loss_weight']
        self.identity_weight = args['identity_loss_weight']
        self.G_lr_scheduler = module.LinearDecay(args['lr'], args['epochs'] * len_dataset, args['epoch_decay'] * len_dataset)
        self.D_lr_scheduler = module.LinearDecay(args['lr'], args['epochs'] * len_dataset, args['epoch_decay'] * len_dataset)
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=self.G_lr_scheduler, beta_1=args['beta_1'])
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=self.D_lr_scheduler, beta_1=args['beta_1'])

        # Creating models.
        
        self.set_G_A2B(input_shape=(args['crop_size'], args['crop_size'], 3), q = args['q'])
        self.set_G_B2A(input_shape=(args['crop_size'], args['crop_size'], 3), q = args['q'])
        self.set_D_A(input_shape=(args['crop_size'], args['crop_size'], 3), q = args['q'])
        self.set_D_B(input_shape=(args['crop_size'], args['crop_size'], 3), q = args['q'])

    def set_G_A2B(self, input_shape, q):
        if self.filter == 'operational':
            self.G_A2B = module.OpGenerator(input_shape = input_shape, q = q)
        elif self.filter == 'convolutional':
            self.G_A2B = module.ConvGenerator(input_shape = input_shape)
        elif self.filter == 'convolutional-light':
            self.G_A2B = module.ConvCompGenerator(input_shape = input_shape)
        else: print('Undefined filtering method!')

        
    def set_G_B2A(self, input_shape, q):
        if self.filter == 'operational':
            self.G_B2A = module.OpGenerator(input_shape = input_shape, q = q)
        elif self.filter == 'convolutional':
            self.G_B2A = module.ConvGenerator(input_shape = input_shape)
        elif self.filter == 'convolutional-light':
            self.G_B2A = module.ConvCompGenerator(input_shape = input_shape)
        else: print('Undefined filtering method!')

    def set_D_A(self, input_shape, q):
        if self.filter == 'operational':
            self.D_A = module.OpDiscriminator(input_shape = input_shape, q = q)
        elif self.filter == 'convolutional':
            self.D_A = module.ConvDiscriminator(input_shape = input_shape)
        elif self.filter == 'convolutional-light':
            self.D_A = module.ConvCompDiscriminator(input_shape = input_shape)
        else: print('Undefined filtering method!')
        
    def set_D_B(self, input_shape, q):
        if self.filter == 'operational':
            self.D_B = module.OpDiscriminator(input_shape = input_shape, q = q)
        elif self.filter == 'convolutional':
            self.D_B = module.ConvDiscriminator(input_shape = input_shape)
        elif self.filter == 'convolutional-light':
            self.D_B = module.ConvCompDiscriminator(input_shape = input_shape)
        else: print('Undefined filtering method!')

    @tf.function
    def train_G(self, A, B):
        with tf.GradientTape() as t:
            
            A2B, y_A2B = self.G_A2B(A[0], training=True) # label_A
            B2A, y_B2A = self.G_B2A(B[0], training=True) # label_B
            A2B2A, y_A2B2A = self.G_B2A(A2B, training=True) # label_A
            B2A2B, y_B2A2B = self.G_A2B(B2A, training=True) # label_B
            A2A, y_A2A = self.G_B2A(A[0], training=True) # label_A
            B2B, y_B2B = self.G_A2B(B[0], training=True) # label_B

            A2B_d_logits = self.D_B(A2B, training=True)
            B2A_d_logits = self.D_A(B2A, training=True)

            A2B_g_loss = self.g_loss_fn(A2B_d_logits)
            B2A_g_loss = self.g_loss_fn(B2A_d_logits)
            A2B2A_cycle_loss = self.cycle_loss_fn(A[0], A2B2A)
            B2A2B_cycle_loss = self.cycle_loss_fn(B[0], B2A2B)
            A2A_id_loss = self.identity_loss_fn(A[0], A2A)
            B2B_id_loss = self.identity_loss_fn(B[0], B2B)

            # Classification losses.
            A2B_c_loss = self.class_loss_fn(A[1], y_A2B) # label_A
            A2B2A_c_loss = self.class_loss_fn(A[1], y_A2B2A) # label_A
            A2A_c_loss = self.class_loss_fn(A[1], y_A2A) # label_A
            B2A_c_loss = self.class_loss_fn(B[1], y_B2A) # label_B
            B2A2B_c_loss = self.class_loss_fn(B[1], y_B2A2B) # label_B
            B2B_c_loss = self.class_loss_fn(B[1], y_B2B) # label_B

            G_loss = (A2B_g_loss + B2A_g_loss + (0.1 * (A2B_c_loss + B2A_c_loss))) + (
                        A2B2A_cycle_loss + B2A2B_cycle_loss + (0.01 * (A2B2A_c_loss + B2A2B_c_loss))) * self.cycle_weights + (
                        A2A_id_loss + B2B_id_loss + (0.02 * (A2A_c_loss + B2B_c_loss))) * self.identity_weight

        G_grad = t.gradient(G_loss, self.G_A2B.trainable_variables + self.G_B2A.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grad, self.G_A2B.trainable_variables + self.G_B2A.trainable_variables))

        return A2B, B2A

    @tf.function
    def train_D(self, A, B, A2B, B2A):
        with tf.GradientTape() as t:
            A_d_logits = self.D_A(A, training=True)
            B2A_d_logits = self.D_A(B2A, training=True)
            B_d_logits = self.D_B(B, training=True)
            A2B_d_logits = self.D_B(A2B, training=True)

            A_d_loss, B2A_d_loss = self.d_loss_fn(A_d_logits, B2A_d_logits)
            B_d_loss, A2B_d_loss = self.d_loss_fn(B_d_logits, A2B_d_logits)
            
            D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss)

        D_grad = t.gradient(D_loss, self.D_A.trainable_variables + self.D_B.trainable_variables)
        self.D_optimizer.apply_gradients(zip(D_grad, self.D_A.trainable_variables + self.D_B.trainable_variables))


    @tf.function
    def sample(self, A, B):
        A2B, _ = self.G_A2B(A, training=False)
        B2A, _ = self.G_B2A(B, training=False)
        A2B2A, _ = self.G_B2A(A2B, training=False)
        B2A2B, _ = self.G_A2B(B2A, training=False)
        return A2B, B2A, A2B2A, B2A2B