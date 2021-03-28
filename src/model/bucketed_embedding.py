import torch

class BucketedEmbedding(torch.nn.Embedding):

    def __init__(self, bucket_size, num_embeddings, *args, **kwargs):
        self.bucket_size = bucket_size
        real_num_embeddings = (num_embeddings + bucket_size - 1) // bucket_size
        super(BucketedEmbedding, self).__init__(real_num_embeddings, *args, **kwargs)

    def forward(self, indices):
        d = indices.div(self.bucket_size).type(torch.LongTensor)
        return super(BucketedEmbedding, self).forward(d.cuda() if indices.is_cuda else d)
