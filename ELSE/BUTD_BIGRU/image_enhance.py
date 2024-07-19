import torch

######################## classic
def forward(self, image, image_lengths):
    # Extract image feature vectors.
    features = self.fc(image)                  # (bs,36,1024)
    if self.precomp_enc_type == 'basic':
        # When using pre-extracted region features, add an extra MLP for embedding transformation
        features = self.mlp(image) + features  # (bs,36,1024)
    if self.training:
        features_external1 = self.linear1(features)
        features_external2 = self.linear1(features)
        feature_img1 = torch.mean(features_external1, dim=1)
        feature_img2 = torch.mean(features_external2, dim=1)
        feature_img = torch.cat((feature_img1.unsqueeze(1), feature_img2.unsqueeze(1)), dim=1).reshape(-1, features.size(-1))
    else:
        features_external1 = self.linear1(features)
        feature_img = torch.mean(features_external1, dim=1)
        feature_img = feature_img
    feature_img = l2norm(feature_img,dim=-1)
    return feature_img


######################## row
def forward(self, image, image_lengths):
    # Extract image feature vectors.
    features = self.fc(image)
    if self.precomp_enc_type == 'basic':
        # When using pre-extracted region features, add an extra MLP for embedding transformation
        features = self.mlp(image) + features

    if self.training:
        img_emb = features
        features_external = self.linear1(features)

        rand_list_1 = torch.rand(features.size(0), features.size(1)).to(features.device)
        rand_list_2 = torch.rand(features.size(0), features.size(1)).to(features.device)
        mask1 = (rand_list_1 >= 0.2).unsqueeze(-1)
        mask2 = (rand_list_2 >= 0.2).unsqueeze(-1)

        # mask1
        features_external1 = features_external.masked_fill(mask1 == 0, -10000)
        features_k_softmax1 = nn.Softmax(dim=1)(
            features_external1 - torch.max(features_external1, dim=1)[0].unsqueeze(1))
        attn1 = features_k_softmax1.masked_fill(mask1 == 0, 0)
        feature_img1 = torch.sum(attn1 * img_emb, dim=1)
        # mask2
        features_external2 = features_external.masked_fill(mask2 == 0, -10000)
        features_k_softmax2 = nn.Softmax(dim=1)(
            features_external2 - torch.max(features_external2, dim=1)[0].unsqueeze(1))
        attn2 = features_k_softmax2.masked_fill(mask2 == 0, 0)
        feature_img2 = torch.sum(attn2 * img_emb, dim=1)
        feature_img = torch.cat((feature_img1.unsqueeze(1), feature_img2.unsqueeze(1)), dim=1).reshape(-1,img_emb.size(-1))

    else:
        img_emb = features
        features_in = self.linear1(features)
        attn = nn.Softmax(dim=1)(features_in - torch.max(features_in, dim=1)[0].unsqueeze(1))
        feature_img = torch.sum(attn * img_emb, dim=1)

    if not self.no_imgnorm:
        feature_img = l2norm(feature_img, dim=-1)

    return feature_img

######################## VisualSA+image_esa(global+hal)
def forward(self, image, image_lengths):
    """Extract image feature vectors."""
    # assuming that the precomputed features are already l2-normalized
    img = self.fc(image)
    # normalize in the joint embedding space
    if not self.no_imgnorm:
        img = l2norm(img, dim=-1)

    if self.training:
        row_global = torch.mean(img, 1)
        new_global = self.v_global_w(img, row_global)  # new_global(batch_size,1024)

        features = self.fc(image)
        features = self.mlp(image) + features
        img_emb = features
        features_external = self.linear1(features)
        rand_list_1 = torch.rand(features.size(0), features.size(1)).to(features.device)
        mask1 = (rand_list_1 >= 0.2).unsqueeze(-1)
        # mask1
        features_external1 = features_external.masked_fill(mask1 == 0, -10000)
        features_k_softmax1 = nn.Softmax(dim=1)(
            features_external1 - torch.max(features_external1, dim=1)[0].unsqueeze(1))
        attn1 = features_k_softmax1.masked_fill(mask1 == 0, 0)
        feature_img1 = torch.sum(attn1 * img_emb, dim=1)
        feature_img = torch.cat((new_global.unsqueeze(1), feature_img1.unsqueeze(1)), dim=1).reshape(-1,img.size(-1))
    else:
        row_global = torch.mean(img, 1)
        feature_img = self.v_global_w(img, row_global)

    return feature_img

######################## double globals
def forward(self, image, image_lengths):
    """Extract image feature vectors."""
    # assuming that the precomputed features are already l2-normalized
    img = self.fc(image)
    # normalize in the joint embedding space
    if not self.no_imgnorm:
        img = l2norm(img, dim=-1)

    if self.training:
        row_global = torch.mean(img, 1)
        new_global = self.v_global_w(img, row_global)  # new_global(batch_size,1024)
        new_global1 = self.v_global_w(img, row_global)
        feature_img = torch.cat((new_global.unsqueeze(1), new_global1.unsqueeze(1)), dim=1).reshape(-1, img.size(-1))
    else:
        row_global = torch.mean(img, 1)
        feature_img = self.v_global_w(img, row_global)

    feature_img = l2norm(feature_img, dim=-1)

    return feature_img

######################## esa+weights
def forward(self, image, image_lengths):
    # Extract image feature vectors.
    features = self.fc(image)

    if self.precomp_enc_type == 'basic':
        # When using pre-extracted region features, add an extra MLP for embedding transformation
        features = self.mlp(image) + features  # (128,36,1024)

    if not self.no_imgnorm:
        img = l2norm(features, dim=-1)
        # calculate weights
        row_global = torch.mean(img, 1)
        weights = self.v_global_w(img, row_global)  # new_global(batch_size,1024)

    if self.training:
        # calculate attention matrix
        img_emb = features                                          # V
        features_external = self.linear1(features)                  # Q

        # mask1
        rand_list_1 = torch.rand(features.size(0), features.size(1)).to(features.device)
        mask1 = (rand_list_1 >= 0.2).unsqueeze(-1)
        features_external1 = features_external.masked_fill(mask1 == 0, -10000)
        features_k_softmax1 = nn.Softmax(dim=1)(features_external1 - torch.max(features_external1, dim=1)[0].unsqueeze(1))
        attn1 = features_k_softmax1.masked_fill(mask1 == 0, 0)
        score1 = attn1 * img_emb                                    # 注意力分数/权重：(128,36,1024)
        # feature_img = score * weight                              # 赋予权重：(128,36,1024)
        feature_img_weight = (weights.unsqueeze(2) * score1).sum(dim=1)
        feature_img_weight = torch.sum(feature_img_weight, dim=1)   # (128,1024)
        feature_img1 = l2norm(feature_img_weight, dim=-1)           # (128,1024)

        # mask2
        rand_list_2 = torch.rand(features.size(0), features.size(1)).to(features.device)
        mask2 = (rand_list_2 >= 0.2).unsqueeze(-1)
        features_external2 = features_external.masked_fill(mask2 == 0, -10000)
        features_k_softmax2 = nn.Softmax(dim=1)(features_external2 - torch.max(features_external2, dim=1)[0].unsqueeze(1))
        attn2 = features_k_softmax2.masked_fill(mask2 == 0, 0)
        score2 = attn2 * img_emb                                    # 注意力分数/权重：(128,36,1024)
        # feature_img = score * weight                              # 赋予权重：(128,36,1024)
        feature_img_weight = (weights.unsqueeze(2) * score2).sum(dim=1)
        feature_img_weight = torch.sum(feature_img_weight, dim=1)   # (128,1024)
        feature_img2 = l2norm(feature_img_weight, dim=-1)           # (128,1024)

        feature_img = torch.cat((feature_img1.unsqueeze(1), feature_img2.unsqueeze(1)), dim=1).reshape(-1, img_emb.size(-1))

    else:
        img_emb = features
        features_in = self.linear1(features)
        attn = nn.Softmax(dim=1)(features_in - torch.max(features_in, dim=1)[0].unsqueeze(1))
        score = attn * img_emb
        feature_img_weight = (weights.unsqueeze(2) * score).sum(dim=1)
        feature_img = torch.sum(feature_img_weight, dim=1)

    if not self.no_imgnorm:
        feature_img = l2norm(feature_img, dim=-1)

    return feature_img