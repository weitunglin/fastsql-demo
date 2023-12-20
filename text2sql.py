import os
import json
import torch
import copy
import re
import sqlparse
import sqlite3
from functools import partial

from tqdm import tqdm
from utils.db_utils import get_db_schema
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from pyserini.search.lucene import LuceneSearcher
from utils.db_utils import check_sql_executability, get_matched_contents, get_db_schema_sequence, get_matched_content_sequence, get_db_schema_sequence_natsql
from schema_item_filter import SchemaItemClassifierInference, filter_schema
from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora, get_lora_state_dict, LoRAParametrization

from natsql_utils import natsql_to_sql

def remove_similar_comments(names, comments):
    '''
    Remove table (or column) comments that have a high degree of similarity with their names

    Arguments:
        names: a list of table (or column) names
        comments: a list of table (or column) comments

    Returns:
        new_comments: a list of new table (or column) comments
    '''
    new_comments = []
    for name, comment in zip(names, comments):
        if name.replace("_", "").replace(" ", "") == comment.replace("_", "").replace(" ", ""):
            new_comments.append("")
        else:
            new_comments.append(comment)

    return new_comments

def load_db_comments(table_json_path):
    additional_db_info = json.load(open(table_json_path))
    db_comments = dict()
    for db_info in additional_db_info:
        comment_dict = dict()

        column_names = [column_name.lower() for _, column_name in db_info["column_names_original"]]
        table_idx_of_each_column = [t_idx for t_idx, _ in db_info["column_names_original"]]
        column_comments = [column_comment.lower() for _, column_comment in db_info["column_names"]]

        assert len(column_names) == len(column_comments)
        column_comments = remove_similar_comments(column_names, column_comments)

        table_names = [table_name.lower() for table_name in db_info["table_names_original"]]
        table_comments = [table_comment.lower() for table_comment in db_info["table_names"]]

        assert len(table_names) == len(table_comments)
        table_comments = remove_similar_comments(table_names, table_comments)

        # enumerate each table and its columns
        for table_idx, (table_name, table_comment) in enumerate(zip(table_names, table_comments)):
            comment_dict[table_name] = {
                "table_comment": table_comment,
                "column_comments": dict()
            }
            for t_idx, column_name, column_comment in zip(table_idx_of_each_column, column_names, column_comments):
                # record columns in current table
                if t_idx == table_idx:
                    comment_dict[table_name]["column_comments"][column_name] = column_comment

        db_comments[db_info["db_id"]] = comment_dict

    return db_comments

def get_db_id2schema(db_path, tables_json):
    db_comments = load_db_comments(tables_json)
    db_id2schema = dict()

    for db_id in tqdm(os.listdir(db_path)):
        db_id2schema[db_id] = get_db_schema(os.path.join(db_path, db_id, db_id + ".sqlite"), db_comments, db_id)

    return db_id2schema

def get_db_id2ddl(db_path):
    db_ids = os.listdir(db_path)
    db_id2ddl = dict()

    for db_id in db_ids:
        conn = sqlite3.connect(os.path.join(db_path, db_id, db_id + ".sqlite"))
        cursor = conn.cursor()
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        ddl = []

        for table in tables:
            table_name = table[0]
            table_ddl = table[1]
            table_ddl.replace("\t", " ")
            while "  " in table_ddl:
                table_ddl = table_ddl.replace("  ", " ")

            # remove comments
            table_ddl = re.sub(r'--.*', '', table_ddl)

            table_ddl = sqlparse.format(table_ddl, keyword_case = "upper", identifier_case = "lower", reindent_aligned = True)
            table_ddl = table_ddl.replace(", ", ",\n    ")

            if table_ddl.endswith(";"):
                table_ddl = table_ddl[:-1]
            table_ddl = table_ddl[:-1] + "\n);"
            table_ddl = re.sub(r"(CREATE TABLE.*?)\(", r"\1(\n    ", table_ddl)

            ddl.append(table_ddl)
        db_id2ddl[db_id] = "\n\n".join(ddl)

    return db_id2ddl

class ChatBot():
    def __init__(self, use_fastsql=True) -> None:
        #os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        #model_name = "seeklhy/codes-3b"

        self.use_fastsql = use_fastsql

        model_name = '/workspace/resdsql/models/text2natsql-t5-base/checkpoint-14352/'
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto", torch_dtype = torch.float16)

        self.tokenizer = T5TokenizerFast.from_pretrained(model_name, add_prefix_space=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

        self.max_length = 2048
        self.max_new_tokens = 1024
        self.max_prefix_length = self.max_length - self.max_new_tokens

        self.sic = SchemaItemClassifierInference("sic_ckpts/sic_spider")

        self.db_id2content_searcher = dict()
        for db_id in os.listdir("db_contents_index"):
            self.db_id2content_searcher[db_id] = LuceneSearcher(os.path.join("db_contents_index", db_id))

        self.db_ids = sorted(os.listdir("databases"))
        self.db_id2schema = get_db_id2schema("databases", "data/tables.json")
        self.db_id2ddl = get_db_id2ddl("databases")

        tables = json.load(open('/workspace/resdsql/dog_kennels/preprocessed_data/tables_for_natsql.json', 'r'))
        self.table_dict = dict()
        for t in tables:
            self.table_dict[t["db_id"]] = t


        if not self.use_fastsql:
            return

        self.lora_dbs = {
            'dog_kennels': '/workspace/resdsql/models/text2natsql-t5-base-dog-kennels-lora/checkpoint-651.lora',
            'cars_1': '/workspace/resdsql/models/text2natsql-t5-base-car-lora/checkpoint-912.lora'
        }

        lora_state_dicts = []
        lora_config = {  # specify which layers to add lora to, by default only add to linear layers
            torch.nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=128, lora_alpha=128, lora_dropout_p=0.05),
            },
        }
        self.model.cpu()
        add_lora(self.model, lora_config)
        self.model.cuda()
        for lora in self.lora_dbs.keys():
            print(self.model.device)
            state_dict = torch.load(self.lora_dbs[lora], map_location=self.model.device)
            lora_state_dicts.append(state_dict)

        load_multiple_lora(self.model, lora_state_dicts)

    def get_response_natsql(self, question, db_id):
        print(self.model.device)

        data = { "text": question, "schema": copy.deepcopy(self.db_id2schema[db_id]) }
        data = filter_schema(data, self.sic, 6, 10)
        data["schema_sequence"] = get_db_schema_sequence_natsql(data["schema"])

        prefix_seq = data["text"] + data["schema_sequence"]
        print('prefix seq')
        print(prefix_seq)

        input_ids = self.tokenizer(prefix_seq , truncation = False)["input_ids"]
        attention_mask = [1] * len(input_ids)

        inputs = {
            "input_ids": torch.tensor([input_ids], dtype = torch.int64).to(self.model.device),
            "attention_mask": torch.tensor([attention_mask], dtype = torch.int64).to(self.model.device)
        }
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():

            if self.use_fastsql and db_id in self.lora_dbs.keys():
                m = select_lora(self.model, list(self.lora_dbs.keys()).index(db_id))
            else:
                m = self.model

            print(m.device)

            generate_ids = self.model.generate(
                **inputs,
                decoder_start_token_id = self.model.config.decoder_start_token_id,
                # max_new_tokens = self.max_new_tokens,
                max_length = 256,
                num_beams = 8,
                num_return_sequences = 8
            )

        # print(generate_ids.shape)
        generated_sqls = self.tokenizer.batch_decode(generate_ids[:, :], skip_special_tokens = True)
        # print(generated_sqls)

        final_generated_sql = None
        for generated_sql in generated_sqls:
            generated_sql = generated_sql.split("|")[-1].strip()
            generated_sql = generated_sql.replace("='", "= '").replace("!=", " !=").replace(",", " ,")

            pred_sql = natsql_to_sql(generated_sql, db_id, os.path.join("databases", db_id, db_id + ".sqlite"), self.table_dict[db_id]).strip()
            execution_error = check_sql_executability(generated_sql, os.path.join("databases", db_id, db_id + ".sqlite"))
            if execution_error is None: # the generated sql has no execution errors, we will return it as the final generated sql
                final_generated_sql = generated_sql
                break

        if final_generated_sql is None:
            if generated_sqls[0].split("|")[-1].strip() != "":
                final_generated_sql = generated_sqls[0].split("|")[-1].strip()
            else:
                final_generated_sql = "Sorry, I can not generate a suitable SQL query for your question."

        return final_generated_sql.replace("\n", " ")

    def get_response(self, question, db_id):
        data = {
            "text": question,
            "schema": copy.deepcopy(self.db_id2schema[db_id]),
            "matched_contents": get_matched_contents(question, self.db_id2content_searcher[db_id])
        }
        # print(data)
        data = filter_schema(data, self.sic, 6, 10)
        data["schema_sequence"] = get_db_schema_sequence(data["schema"])
        # data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])

        # prefix_seq = data["schema_sequence"] + "\n" + data["content_sequence"] + "\n" + data["text"] + "\n"
        prefix_seq = data["schema_sequence"] + "\n" + data["text"] + "\n"
        print('prefix seq')
        print(prefix_seq)

        #input_ids = [self.tokenizer.bos_token_id] + self.tokenizer(prefix_seq , truncation = False)["input_ids"]

        input_ids = self.tokenizer(prefix_seq , truncation = False)["input_ids"]
        # print(input_ids)

        if len(input_ids) > self.max_prefix_length:
            print("the current input sequence exceeds the max_tokens, we will truncate it.")
            # input_ids = [self.tokenizer.bos_token_id] + input_ids[-(self.max_prefix_length-1):]
            input_ids = input_ids[-(self.max_prefix_length-1):]
        attention_mask = [1] * len(input_ids)

        inputs = {
            "input_ids": torch.tensor([input_ids], dtype = torch.int64).to(self.model.device),
            "attention_mask": torch.tensor([attention_mask], dtype = torch.int64).to(self.model.device)
        }
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens,
                num_beams = 8,
                num_return_sequences = 8
            )

        generated_sqls = self.tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)
        print(generated_sqls)
        final_generated_sql = None
        for generated_sql in generated_sqls:
            execution_error = check_sql_executability(generated_sql, os.path.join("databases", db_id, db_id + ".sqlite"))
            if execution_error is None: # the generated sql has no execution errors, we will return it as the final generated sql
                final_generated_sql = generated_sql
                break

        if final_generated_sql is None:
            if generated_sqls[0].strip() != "":
                final_generated_sql = generated_sqls[0].strip()
            else:
                final_generated_sql = "Sorry, I can not generate a suitable SQL query for your question."

        return final_generated_sql.replace("\n", " ")
