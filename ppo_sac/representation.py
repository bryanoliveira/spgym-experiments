import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import models

def add_args(parser):
    # Representation learning flags
    parser.add_argument("--use-reward-model", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-reward-loss", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-transition-model", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-transition-loss", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-curl-loss", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-reconstruction-loss", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-data-augmentation-loss", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-dbc-loss", type=bool, action=argparse.BooleanOptionalAction, default=False, help="state metric loss")
    parser.add_argument("--use-spr-loss", type=bool, action=argparse.BooleanOptionalAction, default=False, help="non-contrastive self-prediction loss")

    # Training hyperparameters
    # General
    parser.add_argument("--repr-learning-rate", type=float, default=1e-3)
    parser.add_argument("--repr-loss-coef", type=float, default=1)
    parser.add_argument("--repr-decay-weight", type=float, default=0)  # 1e-6 Yarats 2021
    parser.add_argument("--independent-encoder", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target-encoder-frequency", type=int, default=1)
    parser.add_argument("--encoder-tau", type=float, default=0.05)
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    # Decoder & reconstruction
    parser.add_argument("--train-decoder", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decoder-decay-weight", type=float, default=0)  # 1e-7 Yarats 2021
    parser.add_argument("--variational-reconstruction", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--variational-kl-weight", type=float, default=1e-7)  # Yarats 2021
    # Data augmentation & contrastive learning
    parser.add_argument("--apply-data-augmentation", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augmentations", type=str, default="grayscale,channel_shuffle")
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--contrastive-positives", type=str, default="temporal", choices=["temporal", "augmented"])
    # SPR -- use batch_size 3500 for spr
    parser.add_argument("--spr-horizon", type=int, default=3)
    # Transition model
    parser.add_argument("--probabilistic-transition-model", type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-transition-model-sigma", type=float, default=1e-4)
    parser.add_argument("--max-transition-model-sigma", type=float, default=1e1)

    # Functional args
    parser.add_argument("--repr-update-frequency", type=int, default=1)
    parser.add_argument("--repr-eval-every", type=int, default=int(1e3))
    parser.add_argument("--repr-eval-samples", type=int, default=64)
    

class Representation:
    def __init__(self, args, action_space, encoder, target_encoder, writer, device):
        self.disabled = False
        self.args = args
        self.writer = writer
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.num_updates = 0
        self.global_step = 0
        self.next_log = 0
        self.next_eval = 0
        self.eval_obs = None
        self.eval_next_obs = None
        self.eval_actions = None

        self.parameters = [*self.encoder.parameters()]

        if args.train_decoder:
            if self.encoder.visual:
                self.decoder = models.ImageDecoder(
                    n_channels=self.encoder.base_encoder.n_channels,
                    min_res=self.encoder.base_encoder.min_res,
                    hidden_size=args.hidden_size,
                ).to(device)
            else:
                self.decoder = models.StateDecoder(
                    state_size=self.encoder.base_encoder.state_size,
                    hidden_size=args.hidden_size,
                ).to(device)
            self.parameters.extend(self.decoder.parameters())

        if args.use_transition_model:
            self.transition_model = models.TransitionModel(
                hidden_size=args.hidden_size,
                action_space=action_space,
                probabilistic=args.probabilistic_transition_model,
                min_sigma=args.min_transition_model_sigma,
                max_sigma=args.max_transition_model_sigma
            ).to(device)
            self.parameters.extend(self.transition_model.parameters())

        if args.use_reward_model:
            self.reward_decoder = models.MLP(
                hidden_size=args.hidden_size,
                n_layers=2,
                outputs=1
            ).to(device)
            self.parameters.extend(self.reward_decoder.parameters())

        if args.use_spr_loss:
            self.projector = models.MLP(
                hidden_size=args.hidden_size,
                n_layers=2,
            ).to(device)
            self.target_projector = models.get_target(self.projector)
            self.parameters.extend(self.projector.parameters())

            self.predictor = models.MLP(
                hidden_size=args.hidden_size,
                n_layers=2,
            ).to(device)
            self.parameters.extend(self.predictor.parameters())

        if args.use_curl_loss:
            # don't put it in the gpu yet
            self.curl_W = nn.Parameter(torch.rand(args.hidden_size, args.hidden_size))
            self.parameters.append(self.curl_W)

        self.optimizer = optim.Adam(self.parameters, lr=args.repr_learning_rate)

        if args.use_curl_loss:
            # after creating the optimizer to avoid non-leaf tensor error 
            self.curl_W = self.curl_W.to(device)


    def update(self, data, global_step):
        if self.disabled:
            return
    
        self.num_updates += 1
        self.global_step = global_step

        encoded_obs, mu, logvar = self.encoder(data.observations, return_mu_logvar=True)

        loss = 0

        if self.args.repr_decay_weight > 0:
            repr_decay_loss = self.args.repr_decay_weight * encoded_obs.pow(2).mean()
            loss += repr_decay_loss
            self.log("losses", repr_decay_loss=repr_decay_loss.item())
        # representation losses
        if self.args.use_transition_loss:
            pred_next_state, pred_next_mu, pred_next_sigma = self.transition_model(
                encoded_obs, data.actions, return_mu_sigma=True
            )
            encoded_next_obs = self.target_encoder(data.next_observations)
            effective_sigma = (
                pred_next_sigma
                if pred_next_sigma is not None
                else torch.ones_like(pred_next_mu)
            )
            diff = (pred_next_mu - encoded_next_obs.detach()) / effective_sigma
            transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(effective_sigma))
            loss += transition_loss
            self.log("losses", transition_loss=transition_loss.item())
        if self.args.use_reward_loss:
            decoded_rewards = self.reward_decoder(
                pred_next_state if self.args.use_transition_loss else encoded_next_obs
            )
            reward_loss = F.mse_loss(decoded_rewards.squeeze(), data.rewards.squeeze())
            loss += reward_loss
            self.log("losses", reward_loss=reward_loss.item())
        if self.args.use_curl_loss:
            # from dmcontrol-generalization-benchmark/src/algorithms/curl.py
            """
            Uses logits trick for CURL:
            - compute (B,B) matrix z_a (W z_pos.T)
            - positives are all diagonal elements
            - negatives are all other elements
            - to compute loss use multiclass cross entropy with identity matrix for labels
            """
            x_a = self.encoder(data.observations)
            if self.args.contrastive_positives == "temporal":
                z_pos = self.target_encoder(data.next_observations)
            elif self.args.contrastive_positives == "augmented":
                z_pos = self.target_encoder(data.posterior)

            Wz = torch.matmul(self.curl_W, z_pos.T)  # (z_dim, B)
            logits = torch.matmul(x_a, Wz)  # (B,B)
            logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(logits.shape[0]).long().to(self.curl_W.device)
            contrastive_loss = F.cross_entropy(logits, labels)
            loss += contrastive_loss
            self.log("losses", contrastive_loss=contrastive_loss.item())
        if self.args.use_data_augmentation_loss:
            encoded_posterior = self.target_encoder(data.posterior)
            augmentation_loss = F.mse_loss(encoded_posterior, encoded_obs)
            loss += augmentation_loss
            self.log("losses", augmentation_loss=augmentation_loss.item())
        if self.args.train_decoder or self.args.use_reconstruction_loss:
            decoded_obs = self.decoder(
                encoded_obs if self.args.use_reconstruction_loss else encoded_obs.detach()
            )
            reconstruction_loss = F.mse_loss(
                decoded_obs.reshape(data.observations.shape[0], -1),
                data.observations.reshape(data.observations.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)
            if self.args.variational_reconstruction:
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
                reconstruction_loss += self.args.variational_kl_weight * kl_loss
                self.log("losses", reconstruction_kl_loss=kl_loss.mean().item())
            if self.args.decoder_decay_weight > 0:
                decoder_decay_loss = self.args.decoder_decay_weight * sum(
                    p.pow(2).mean() for p in self.decoder.parameters()
                )
                loss += decoder_decay_loss
                self.log("losses", decoder_decay_loss=decoder_decay_loss.item())
            reconstruction_loss = reconstruction_loss.mean()
            loss += reconstruction_loss
            self.log("losses", reconstruction_loss=reconstruction_loss.item())
        if self.args.use_dbc_loss:
            # Compute the state metric loss inspired by Deep Bisimulation for Control
            # Reference: https://arxiv.org/abs/2006.10742
            # Implemented as close as possible to https://github.com/facebookresearch/deep_bisim4control/blob/main/agent/bisim_agent.py
            # assume transition model was already updated
            # NOTE: only works with probabilistic transition model
            pred_next_mu = pred_next_mu.detach()
            pred_next_sigma = pred_next_sigma.detach()
            permuted_indices = torch.randperm(encoded_obs.size(0))
            state_distances = F.smooth_l1_loss(
                encoded_obs, encoded_obs[permuted_indices], reduction="none"
            ).mean(
                dim=1
            )  # L1
            # Or: state_distances = torch.norm(encoded_obs - encoded_obs[permuted_indices], p=2, dim=1)  # L2
            transition_distances = torch.sqrt(
                (pred_next_mu - pred_next_mu[permuted_indices]).pow(2)
                + (pred_next_sigma - pred_next_sigma[permuted_indices]).pow(2)
            ).mean(
                dim=1
            )  # euclidean distance like the original implementation
            reward_distances = F.smooth_l1_loss(
                data.rewards, data.rewards[permuted_indices], reduction="none"
            )
            bisimilarity = reward_distances - self.args.gamma * transition_distances
            dbc_loss = (state_distances - bisimilarity).pow(2).mean()
            # Or: dbc_loss = F.mse_loss(state_distances, transition_distances)  # theta_loss
            loss += dbc_loss
            self.log("losses", state_metric_loss=dbc_loss.item())
        if self.args.use_spr_loss:
            # Implementing Self-Predictive Representations (SPR) loss
            # Reference: https://arxiv.org/abs/2007.05929
            models.update_target(self.projector, self.target_projector)
            z_hat = encoded_obs  # online representations # 0
            z_tildes = self.target_encoder(
                data.multistep_observations.reshape(-1, *data.observations.shape[1:])
            ).reshape(*data.multistep_observations.shape[0:2], -1)  # target representations
            y_tildes = self.target_projector(
                z_tildes.reshape(-1, z_tildes.shape[-1])
            ).reshape(*data.multistep_observations.shape[0:2], -1)  # target projections
            spr_loss = 0
            for k in range(1, self.args.spr_horizon):
                z_hat = self.transition_model(z_hat, data.multistep_actions[k])  # latent states via transition model
                # z_tilde = z_tildes[k]  # target representations (calculated previously)
                y_hat = self.predictor(self.projector(z_hat))  # projections
                y_tilde = y_tildes[k]  # projections
                spr_loss -= F.cosine_similarity(y_hat, y_tilde, dim=-1, eps=1e-6)  # SPR loss at step k
            spr_loss = spr_loss.mean()
            loss += spr_loss
            self.log("losses", noncontrastive_ss_loss=spr_loss.item())

        if loss != 0:
            loss *= self.args.repr_loss_coef
            self.log("losses", representation_loss=loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters, self.args.max_grad_norm)
            self.optimizer.step()
        elif type(loss) != torch.Tensor:
            # loss is 0 and not a tensor, disable the representation module
            print("Disabling representation module as it was not used")
            self.disabled = True
            return
        
        if self.num_updates >= self.next_eval:
            self.next_eval = self.num_updates + self.args.repr_eval_every
            self.evaluate(data, global_step)
        if self.num_updates >= self.next_log:
            self.next_log = self.num_updates + self.args.log_every

    def evaluate(self, data, global_step):
        if self.disabled:
            return

        if self.eval_obs is None:
            # select first step for all envs
            self.eval_obs = data.observations[:self.args.repr_eval_samples].clone()
            self.eval_next_obs = data.next_observations[:self.args.repr_eval_samples].clone()
            self.eval_actions = data.actions[:self.args.repr_eval_samples].clone()

        with torch.no_grad():
            encoded_obs = self.encoder(self.eval_obs)
            encoded_next_obs = self.encoder(self.eval_next_obs)

        if self.args.use_transition_model:
            with torch.no_grad():
                pred_next_state = self.transition_model(encoded_obs, self.eval_actions)
                if hasattr(self, "decoder"):
                    decoded_pred_next = self.decoder(pred_next_state)
                    pred_next_loss = F.mse_loss(decoded_pred_next, self.eval_next_obs)
                    self.log("reconstruction", bypass=True,
                        pred_next_mse=pred_next_loss.item(),
                    )
                    if len(self.eval_obs.shape) == 4:  # an image
                        if self.eval_obs.shape[1] == 4:  # framestacked, plot just one frame
                            decoded_pred_next = decoded_pred_next[:, 0:1] 
                        self.writer.add_image(
                            "reconstruction/decoded_pred_next",
                            decoded_pred_next.permute(0, 2, 3, 1).clip(0, 1).mul(255).byte(),
                            global_step,
                            dataformats="NHWC",
                        )

                flat_pred_next = pred_next_state.reshape(-1, pred_next_state.shape[-1])
                self.writer.add_embedding(
                    flat_pred_next.detach().cpu(),
                    tag="predicted_next_states",
                    global_step=global_step,
                )
            
        if hasattr(self, "decoder"):
            decoded_obs = self.decoder(encoded_obs)
            decoded_next_obs = self.decoder(encoded_next_obs)
            # Calculate and log reconstruction error metrics
            recon_loss = F.mse_loss(decoded_obs, self.eval_obs)
            next_recon_loss = F.mse_loss(decoded_next_obs, self.eval_next_obs)
            self.log("reconstruction",
                obs_mse=recon_loss.item(),
                next_obs_mse=next_recon_loss.item(),
            )
            
            if len(self.eval_obs.shape) == 4:  # an image
                if self.eval_obs.shape[1] > 3:  # framestacked, plot just one frame
                    decoded_obs = decoded_obs[:, 0:1]  # Get first frame from stack
                    self.eval_obs = self.eval_obs[:, 0:1]
                    self.eval_next_obs = self.eval_next_obs[:, 0:1]
                    decoded_next_obs = decoded_next_obs[:, 0:1]

                self.writer.add_image(
                    "reconstruction/decoded",
                    decoded_obs.permute(0, 2, 3, 1).clip(0, 1).mul(255).byte(),
                    global_step,
                    dataformats="NHWC",
                )
                self.writer.add_image(
                    "reconstruction/actual",
                    self.eval_obs.permute(0, 2, 3, 1).clip(0, 1).mul(255).byte(),
                    global_step,
                    dataformats="NHWC",
                )
                
                self.writer.add_image(
                    "reconstruction/decoded_next",
                    decoded_next_obs.permute(0, 2, 3, 1).clip(0, 1).mul(255).byte(),
                    global_step,
                    dataformats="NHWC",
                )
                self.writer.add_image(
                    "reconstruction/actual_next",
                    self.eval_next_obs.permute(0, 2, 3, 1).clip(0, 1).mul(255).byte(),
                    global_step,
                    dataformats="NHWC",
                )
        
        # compute the average difference between encoder and target encoder parameters
        if self.target_encoder is not None:
            assert sum(p.numel() for p in self.target_encoder.parameters() if p.requires_grad) == 0
            encoder_diff = 0
            n_params = 0
            for param, target_param in zip(
                self.encoder.parameters(), self.target_encoder.parameters()
            ):
                encoder_diff += torch.abs(param - target_param).mean().item()
                n_params += 1
            encoder_diff /= n_params
            self.log("encoder", bypass=True,
                target_diff=encoder_diff,
            )

        # flatten the batch dimension for visualization
        flat_encoded_obs = encoded_obs.reshape(-1, encoded_obs.shape[-1])
        flat_encoded_next = encoded_next_obs.reshape(-1, encoded_next_obs.shape[-1])
        # log the embeddings
        self.writer.add_embedding(
            flat_encoded_obs.detach().cpu(),
            tag="encoded_observations",
            global_step=global_step,
        )
        self.writer.add_embedding(
            flat_encoded_next.detach().cpu(),
            tag="encoded_next_observations",
            global_step=global_step,
        )
        # compute distances between encoded states and next states
        distances = torch.norm(flat_encoded_obs - flat_encoded_next, dim=1)
        # compute distances from origin
        origin_distances = torch.norm(flat_encoded_obs, dim=1)
        self.log("embeddings", bypass=True,
            avg_state_next_distance=distances.mean().item(),
            median_state_next_distance=distances.median().item(),
            min_state_next_distance=distances.min().item(),
            max_state_next_distance=distances.max().item(),
            avg_origin_distance=origin_distances.mean().item(),
            median_origin_distance=origin_distances.median().item(),
            min_origin_distance=origin_distances.min().item(),
            max_origin_distance=origin_distances.max().item(),
        )

    def log(self, group, bypass=False, **kwargs):
        if self.num_updates >= self.next_log or bypass:
            for name, value in kwargs.items():
                self.writer.add_scalar(f"{group}/{name}", value, self.global_step)

    def state_dict(self):
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'encoder': self.encoder.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
        }
        if hasattr(self, 'decoder'):
            state_dict['decoder'] = self.decoder.state_dict()
        if hasattr(self, 'transition_model'):
            state_dict['transition_model'] = self.transition_model.state_dict()
        if hasattr(self, 'reward_decoder'):
            state_dict['reward_decoder'] = self.reward_decoder.state_dict()
        if hasattr(self, 'projector'):
            state_dict['projector'] = self.projector.state_dict()
        if hasattr(self, 'target_projector'):
            state_dict['target_projector'] = self.target_projector.state_dict()
        if hasattr(self, 'predictor'):
            state_dict['predictor'] = self.predictor.state_dict()
        if hasattr(self, 'curl_W'):
            state_dict['curl_W'] = self.curl_W
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.encoder.load_state_dict(state_dict['encoder'])
        self.target_encoder.load_state_dict(state_dict['target_encoder'])

        if hasattr(self, 'decoder'):
            self.decoder.load_state_dict(state_dict['decoder'])
        if hasattr(self, 'transition_model'):
            self.transition_model.load_state_dict(state_dict['transition_model'])
        if hasattr(self, 'reward_decoder'):
            self.reward_decoder.load_state_dict(state_dict['reward_decoder'])
        if hasattr(self, 'projector'):
            self.projector.load_state_dict(state_dict['projector'])
        if hasattr(self, 'target_projector'):
            self.target_projector.load_state_dict(state_dict['target_projector'])
        if hasattr(self, 'predictor'):
            self.predictor.load_state_dict(state_dict['predictor'])
        if hasattr(self, 'curl_W'):
            self.curl_W = state_dict['curl_W']