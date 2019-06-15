# the relation consensus module by Bolei
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pdb

class RelationModule(torch.nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.classifier = self.fc_fusion()
    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 512
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck,self.num_class),
                )
        return classifier
    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = self.classifier(input)
        return input

class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(num_bottleneck, self.num_class),
                        )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))




class RelationModuleMultiScaleWithClassifier(torch.nn.Module):
    # relation module in multi-scale with a classifier at the end
    def __init__(self, img_feature_dim, num_frames, num_class):
        print("RelationModuleMultiScaleWithClassifier init")
        super(RelationModuleMultiScaleWithClassifier, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] #

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        self.classifier_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]

            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),# this is the newly added thing
                        nn.Linear(num_bottleneck, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),
                        )
            classifier = nn.Linear(num_bottleneck, self.num_class)
            self.fc_fusion_scales += [fc_fusion]
            self.classifier_scales += [classifier]
        # maybe we put another fc layer after the summed up results???
        print('Multi-Scale Temporal Relation with classifier in use')
        print(['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        act_all = self.classifier_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = self.classifier_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

class RelationModuleMultiScaleWithClassifier_h_after_sum(torch.nn.Module):
    # might correspond to paper

    # relation module in multi-scale with a classifier at the end
    def __init__(self, img_feature_dim, num_frames, num_class):
        print("RelationModuleMultiScaleWithClassifier_h_after_sum init")
        super(RelationModuleMultiScaleWithClassifier_h_after_sum, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] #

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        self.classifier_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]

            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),# this is the newly added thing
                        nn.Linear(num_bottleneck, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6),
                        )
            classifier = nn.Linear(num_bottleneck, self.num_class)
            self.fc_fusion_scales += [fc_fusion]
            self.classifier_scales += [classifier]
        # maybe we put another fc layer after the summed up results???
        print('Multi-Scale Temporal Relation with classifier in use')
        print(['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        final_all = self.classifier_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                # act_relation = self.classifier_scales[scaleID](act_relation)
                # act_all += act_relation
                act_all = act_all + act_relation
            # final_all += self.classifier_scales[scaleID](act_all)
            final_all  = final_all + self.classifier_scales[scaleID](act_all)

        # return act_all
        return final_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

class RelationModuleMultiScaleLSTM(torch.nn.Module):
    # here, replace 2-layer's LSTM in  RelationModuleMultiScaleWithClassifier_h_after_sum with LSTM

    # relation module in multi-scale with a classifier at the end
    def __init__(self, img_feature_dim, num_frames, num_class):
        print("RelationModuleMultiScaleLSTM init")
        super(RelationModuleMultiScaleLSTM, self).__init__()
        self.lstm_layers = 2
        # self.lstm_layers = 1
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] #

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.num_bottleneck = num_bottleneck
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        self.classifier_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]

            # fc_fusion = nn.Sequential(
            #             nn.ReLU(),
            #             nn.Linear(scale * self.img_feature_dim, num_bottleneck),
            #             nn.ReLU(),
            #             nn.Dropout(p=0.6),# this is the newly added thing
            #             nn.Linear(num_bottleneck, num_bottleneck),
            #             nn.ReLU(),
            #             nn.Dropout(p=0.6),
            #             )
            fc_fusion = nn.LSTM(input_size=self.img_feature_dim,hidden_size= num_bottleneck,num_layers=self.lstm_layers,
                                bias=True,  batch_first=True,dropout=0,  bidirectional=False)

            classifier = nn.Linear(num_bottleneck, self.num_class)
            self.fc_fusion_scales += [fc_fusion]
            self.classifier_scales += [classifier]
        # maybe we put another fc layer after the summed up results???
        print('Multi-Scale Temporal Relation with classifier in use')
        print(['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        # act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        this_batch_size = act_all.size(0)
        act_all = act_all.view(act_all.size(0), self.scales[0] , self.img_feature_dim)
        output,(h,c) = self.fc_fusion_scales[0](act_all)
        act_all = h.view(self.lstm_layers, 1, this_batch_size, self.num_bottleneck)[self.lstm_layers-1,0,:,:]
        act_all=act_all.view(this_batch_size,self.num_bottleneck)

        final_all = self.classifier_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]


                # act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] , self.img_feature_dim)
                # act_relation = self.fc_fusion_scales[scaleID](act_relation)

                this_batch_size = act_relation.size(0)
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID], self.img_feature_dim)
                output, (h, c) = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = h.view(self.lstm_layers, 1, this_batch_size,
                                      self.num_bottleneck)[self.lstm_layers - 1, 0, :, :]
                act_relation = act_relation.view(this_batch_size, self.num_bottleneck)


                # act_all += act_relation
                act_all = act_all + act_relation

            # final_all += self.classifier_scales[scaleID](act_all)
            final_all = final_all + self.classifier_scales[scaleID](act_all)

        # return act_all
        return final_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

class RelationModuleMultiScaleLSTM_RelationModuleMultiScale(torch.nn.Module):
    #  RelationModuleMultiScale is the defualt mode in source code for multi-scale TRN
    # here,repalce 2-layer's MLP in RelationModuleMultiScale with LSTM


    def __init__(self, img_feature_dim, num_frames, num_class,lstm_num_layer):
        print("RelationModuleMultiScaleLSTM_RelationModuleMultiScale init")
        print("lstm_num_layer: ",lstm_num_layer)
        super(RelationModuleMultiScaleLSTM_RelationModuleMultiScale, self).__init__()
        self.lstm_layers = lstm_num_layer
        # self.lstm_layers = 1
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] #

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.num_bottleneck = num_bottleneck
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        # self.classifier_scales = nn.ModuleList()
        self.trans_dim = nn.Linear(num_bottleneck, self.num_class)
        for i in range(len(self.scales)):
            scale = self.scales[i]

            # fc_fusion = nn.Sequential(
            #             nn.ReLU(),
            #             nn.Linear(scale * self.img_feature_dim, num_bottleneck),
            #             nn.ReLU(),
            #             nn.Dropout(p=0.6),# this is the newly added thing
            #             nn.Linear(num_bottleneck, num_bottleneck),
            #             nn.ReLU(),
            #             nn.Dropout(p=0.6),
            #             )
            fc_fusion = nn.LSTM(input_size=self.img_feature_dim,hidden_size= num_bottleneck,num_layers=self.lstm_layers,
                                bias=True,  batch_first=True,dropout=0,  bidirectional=False)

            # classifier = nn.Linear(num_bottleneck, self.num_class)
            self.fc_fusion_scales += [fc_fusion]
            # self.classifier_scales += [classifier]
        # maybe we put another fc layer after the summed up results???
        print('Multi-Scale Temporal Relation with classifier in use')
        print(['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        # act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        this_batch_size = act_all.size(0)
        act_all = act_all.view(act_all.size(0), self.scales[0] , self.img_feature_dim)
        output,(h,c) = self.fc_fusion_scales[0](act_all)
        act_all = h.view(self.lstm_layers, 1, this_batch_size, self.num_bottleneck)[self.lstm_layers-1,0,:,:]
        act_all=act_all.view(this_batch_size,self.num_bottleneck)

        # final_all = self.classifier_scales[0](act_all)
        final_all = self.trans_dim(act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]


                # act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] , self.img_feature_dim)
                # act_relation = self.fc_fusion_scales[scaleID](act_relation)

                this_batch_size = act_relation.size(0)
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID], self.img_feature_dim)
                output, (h, c) = self.fc_fusion_scales[scaleID](act_relation)
                act_relation = h.view(self.lstm_layers, 1, this_batch_size,
                                      self.num_bottleneck)[self.lstm_layers - 1, 0, :, :]
                act_relation = act_relation.view(this_batch_size, self.num_bottleneck)
                act_relation = self.trans_dim(act_relation)


                # act_all += act_relation
                # act_all = act_all + act_relation

                final_all = final_all + act_relation

            # final_all += self.classifier_scales[scaleID](act_all)
            # final_all = final_all + self.classifier_scales[scaleID](act_all)

        # return act_all
        return final_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))



def return_TRN(relation_type, img_feature_dim, num_frames, num_class):
    if relation_type == 'TRN':
        TRNmodel = RelationModule(img_feature_dim, num_frames, num_class)
    elif relation_type == 'TRNmultiscale':
        TRNmodel = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)
    elif relation_type == 'RelationModuleMultiScaleWithClassifier':
        TRNmodel = RelationModuleMultiScaleWithClassifier(img_feature_dim, num_frames, num_class)
    elif relation_type == 'RelationModuleMultiScaleWithClassifier_h_after_sum':
        TRNmodel = RelationModuleMultiScaleWithClassifier_h_after_sum(img_feature_dim, num_frames, num_class)
    elif relation_type == 'MultiScaleLSTM':
        TRNmodel = RelationModuleMultiScaleLSTM(img_feature_dim, num_frames, num_class)
    elif relation_type == "RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer2":
        TRNmodel = RelationModuleMultiScaleLSTM_RelationModuleMultiScale(img_feature_dim, num_frames, num_class,2)
    # elif relation_type == "RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer1":
    elif relation_type == "RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer1" :
        TRNmodel = RelationModuleMultiScaleLSTM_RelationModuleMultiScale(img_feature_dim, num_frames, num_class, 1)
    else:
        raise ValueError('Unknown TRN' + relation_type)


    return TRNmodel

if __name__ == "__main__":
    batch_size = 10
    num_frames = 5
    num_class = 174
    img_feature_dim = 512
    input_var = Variable(torch.randn(batch_size, num_frames, img_feature_dim))
    model = RelationModuleMultiScale(img_feature_dim, num_frames, num_class)
    output = model(input_var)
    print(output)
