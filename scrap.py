

			#print(out_pi)
			#iota = 10**-6
			#out_pi = (out_pi + iota)/sum(out_pi + iota)
			#print(out_pi)
			#clamp_exp = -10
			#out_pi = torch.clamp(out_pi, min=10**clamp_exp, max=(1-10**clamp_exp))



            
			print(agent)
			print('\n\na_one_hot: ', a_one_hot)
			print('r_in: ', r)
			print('out_V: ', out_V.item())
			print('out_pi: ', out_pi)
			#print('out_V: ', out_V.item())
			'''print('out_pi.shape', out_pi.shape)
			print('outputs_pi.shape', outputs_pi.shape)
			print('out_V.shape', out_V.shape)
			print('outputs_V.shape', outputs_V.shape)

			print('out_pi', out_pi)'''




			'''print('clipped:\n\n')
			for p in self.NN.parameters():
				print(p.grad)

			exit()
			print(self.NN.state_dict()['out_pi.weight'])
			print(self.NN.state_dict()['out_pi.weight'].sum())
			print('\n\n')
			print(self.NN.state_dict()['out_pi.weight'])
			print(self.NN.state_dict()['out_pi.weight'].sum())
			exit()'''






            			'''
            			R_tot = R.squeeze().sum().item()
            			r_accum = torch.mm(discount, R.squeeze(dim=1))
            			# entropy (already including its minus sign) is bigger for more uneven dist's.
            			# We want to enforce that (at the
            			# beginning), so we want entropy to stay high. To do this, we have to minimize
            			# its NEGATIVE, like the policy.
            			J = (-torch.log(outputs_pi)*(r_accum - outputs_V.detach())).sum() - entropy.sum()
            			loss = beta_V_loss*(r_accum - outputs_V).pow(2).sum()
            			J.backward(retain_graph=True)
            			loss.backward()'''

            			R_tot = 0.0
            			r_accum = 0.0
            			J_tot = torch.tensor([0.0])
            			V_loss_tot = torch.tensor([0.0])
            			entropy_tot = torch.tensor([0.0])
            			#loss_tot = torch.tensor([0.0])
            			GAE = 0.0
            			tau = 0.3
            			for t in range(N_steps-2, -1, -1):

            				r_accum = R[t] + gamma*r_accum
            				R_tot += R[t].item()

            				# Should it have .detach() here? It's mostly the same NN,
            				# but they also do have slightly diff parts...

            				V_loss = beta_V_loss*(r_accum - outputs_V[t]).pow(2)

            				delta_t = R[t] + gamma*outputs_V[t+1].data - outputs_V[t].data
            				GAE = GAE*gamma*tau + delta_t

            				#J = -torch.log(outputs_pi[t])*(r_accum - outputs_V[t].detach())
            				J = -torch.log(outputs_pi[t])*GAE

            				J_tot += J.squeeze()
            				V_loss_tot += V_loss.squeeze()
            				entropy_tot += -entropy[t].squeeze()
            				# Also not clear if I should be stepping after iteration or no..?
            				#J.backward(retain_graph=True)
            				#loss.backward(retain_graph=True)


            			loss_tot = J_tot + V_loss_tot + entropy_tot
            			self.optimizer.zero_grad()
            			#loss.backward(retain_graph=True)
            			#J.backward()
            			loss_tot.backward()


            			nn.utils.clip_grad_norm_(self.NN.parameters(), 0.5)
            			self.optimizer.step()
