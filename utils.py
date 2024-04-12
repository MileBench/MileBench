import os
from torch.utils.data import Dataset, DataLoader

from workers.model_workers import (
    LLaVA
)

name2worker = {
    'llava-v1.5-7b':LLaVA,
}


def get_worker_class(name):
    return name2worker[name]

class MileBenchDataset(Dataset):
    def __init__(
        self, 
        annotation, 
        task_instructions, 
        img_dir, 
        combine_image=None
    ):
        """
        Initialize the MileBenchDataset class.
        
        Parameters:
            annotation (list): List of annotations.
            task_instructions (dict): Dictionary of task instructions.
            img_dir (str): Directory containing images.
            combine_image (int): Number of combined images.
        """
        self.img_dir = img_dir
        self.annotation = annotation
        self.task_instructions = task_instructions
        self.combine_image = combine_image

    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: Length of the dataset.
        """
        return len(self.annotation)

    def __getitem__(self, index):
        '''
        Get item by index from the dataset.
        If self.combine_image is not None, set different context prompt.
        
        Parameters:
            index (int): Index of the item to retrieve.
        
        Returns:
            dict: Dictionary containing sample information.
        
        {
            'sample_id': 1,
            'raw_img_list': ['/path/to/image1',],
            'context': 'what is the image <ImageHere> about?',
            'response': '',
        }
        '''
        ann = self.annotation[index]

        # Set task instruction
        task_instruction = self.task_instructions[ann['task_instruction_id']]
        context = task_instruction + '\n' + ann['task_instance']['context']
        
        # Set choice_list for multi-choice QA
        if 'choice_list' in ann['task_instance'].keys():
            choice_str = '\nChoice list: \n'
            choice_str += '\n'.join([f'{chr(65+idx)}. {item}' 
                for idx, item in enumerate(ann['task_instance']['choice_list'])])
            choice_str += '\nYour answer is: '
            context += choice_str
            
        # Set prompt
        img_num = len(ann['task_instance']['images_path'])
        if self.combine_image:
            # set different context prompt for combined images
            for i in range(img_num):
                rmv_txt = '{image#%d}'% (i+1)
                rmv_tbl = '{table#%d}'% (i+1)
                context = context.replace(rmv_txt, f'<Image {i+1}> ')
                context = context.replace(rmv_tbl, f'<Image {i+1}> ')
            context = '<ImageHere>'*self.combine_image + '\n' + context
        else:
            for i in range(img_num):
                rmv_txt = '{image#%d}'% (i+1)
                rmv_tbl = '{table#%d}'% (i+1)
                context = context.replace(rmv_txt, '<ImageHere>')
                context = context.replace(rmv_tbl, '<ImageHere>')
            
        # Set images paths
        raw_img_list = []
        if self.combine_image:
            combine_image_str = f'combined_{self.combine_image}_images'
            for p in ann['task_instance'][combine_image_str]:
                img_path = os.path.join(self.img_dir.replace(os.path.basename(self.img_dir), combine_image_str), p)
                raw_img_list.append(img_path)
        else:
            for p in ann['task_instance']['images_path']:
                img_path = os.path.join(self.img_dir, p)
                raw_img_list.append(img_path)
        
        return {
            "sample_id": ann['sample_id'],
            "context": context,
            "raw_img_list": raw_img_list,
            "response": str(ann['response'])
        }

    def collate_fn(self, batch):
        """
        Custom collate function for batching samples.
        
        Parameters:
            batch (list): List of samples.
        
        Returns:
            dict: Dictionary containing batched data.
        """
        batch_data={}
        # Use the default key names
        batch_data['id'] = [sample['sample_id'] for sample in batch]
        batch_data['question'] = [sample['context'] for sample in batch]
        batch_data['image_path'] = [sample['raw_img_list'] for sample in batch]
        batch_data['gt_response'] = [sample['response'] for sample in batch]
        return batch_data


