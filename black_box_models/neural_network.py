import torch
import torch.nn as nn
from torch.optim import optimizer
import os
from transformers import BertModel, BertConfig
import time 
import numpy as np

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, device, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 3
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased', config = config)

        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.device = device
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        last_hidden_state_cls = outputs[0][:, 0, :]

        logits = self.classifier(last_hidden_state_cls)

        return logits
    
    def predict(self, instance):
        """Returns the most likely sequence of tags for a sequence of words in `text_batch`.

        Arguments: 
          text_batch: a tensor containing word ids of size (seq_len, batch_size) 
        Returns:
          tag_batch: a tensor containing tag ids of size (seq_len, batch_size)
        """
        with torch.no_grad():
            logits = self.forward(instance.input_ids, instance.attention_mask)
            #print(logits)
            tag_batch = torch.argmax(logits, axis = -1)
            return tag_batch
    
    def predict_proba(self, instance):
        """Returns the most likely sequence of tags for a sequence of words in `text_batch`.

        Arguments: 
          text_batch: a tensor containing word ids of size (seq_len, batch_size) 
        Returns:
          tag_batch: a tensor containing tag ids of size (seq_len, batch_size)
        """
        with torch.no_grad():
            logits = nn.functional.softmax(self.forward(instance.input_ids, instance.attention_mask), dim = 1)
            return logits

class BertNLITrainer:
    
    def __init__(self,model, optimizer, scheduler, device):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.path_to_save = './black_box_models/saved_models/neural_net_bert'
        
    def train(self, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
        model = self.model
        if os.path.isfile(self.path_to_save):
          print("Loading the saved neural network model")
          self.model.load_state_dict(torch.load(self.path_to_save))
          if evaluation == True:
            _, validation_accuracy = self.evaluate(val_dataloader)
            print (f'Validation accuracy: {validation_accuracy:.4f}')
          return
        print("Start training...\n")
        for epoch_i in range(epochs):

            print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
            print("-"*70)

            t0_epoch, t0_batch = time.time(), time.time()

            total_loss, batch_loss, batch_counts = 0, 0, 0

            model.train()

            for step, batch in enumerate(train_dataloader):
                batch_counts +=1
                b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

                model.zero_grad()

                logits = model(b_input_ids, b_attn_mask)
                loss = self.loss_fn(logits, b_labels)
                batch_loss += loss.item()
                total_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()

                if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    time_elapsed = time.time() - t0_batch

                    print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

            avg_train_loss = total_loss / len(train_dataloader)

            print("-"*70)

            if evaluation == True:

                val_loss, val_accuracy = self.evaluate(val_dataloader)

                time_elapsed = time.time() - t0_epoch

                print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                print("-"*70)
            print("\n")

        torch.save(self.model.state_dict(), self.path_to_save)

        print("Training complete!")


    def evaluate(self, val_dataloader):
        
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        
        model = self.model

        model.eval()

        val_accuracy = []
        val_loss = []

        for batch in val_dataloader:
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                logits = model(b_input_ids, b_attn_mask)

            loss = self.loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            preds = torch.argmax(logits, dim=1).flatten()

            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy
    