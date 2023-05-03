#
#  Copyright (c) 2023 IBM Corp.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

def wikitext_dataset_preprocess(examples):
    def preprocess_wikitext(text):
        wikitext_replacements = [(" 's ", "'s "), ("s ' ", "s' "), (' , ', ', '), (' ( ', ' ('), (' )', ')'),
                                 (' : ', ': '), (' ; ', '; '), (' . ', '. '), (' %', '%'),
                                 (' @-@ ', '-'), (' @.@ ', '.'), (' @,@ ', ',')]
        for original, replacement in wikitext_replacements:
            text = text.replace(original, replacement)
        return text

    def filter_wikitext(source_text):
        return len(source_text.split()) > 50

    source_column = 'text'
    sources = []
    for i in range(len(examples[source_column])):
        source = examples[source_column][i]
        if source is not None and filter_wikitext(source):
            sources.append(preprocess_wikitext(source))

    new_examples = {
        source_column: sources,
    }

    return new_examples


def wikinews_dataset_preprocess(examples):
    def get_main_text(news_text):
        return sorted(news_text.split(":"), key=len)[-1].strip()

    def filter_wikinews(source_text):
        text = get_main_text(source_text)
        return len(text.split()) > 50

    def preprocess_wikinews(text):
        main_text = get_main_text(text)
        main_text = main_text.replace('Pillars of Wikinews writing Writing an article ', '')
        return main_text

    source_column = 'text'
    sources = []
    for source_text in examples[source_column]:
        if source_text is not None and filter_wikinews(source_text):
            sources.append(preprocess_wikinews(source_text))

    new_examples = {
        source_column: sources,
    }

    return new_examples


def bookcorpus_dataset_preprocess(examples):
    def preprocess_bookcorpus(text):
        bookcorpus_replacements = [(" '", "'"), (' , ', ', '), (' .', '.'), (' ?', '?'), (" n't", "n't"),
                                   (' : ', ': '), (' ; ', '; '), (' ( ', ' ('), (' )', ')'), (' %', '%')]
        for original, replacement in bookcorpus_replacements:
            text = text.replace(original, replacement)
        return text

    def filter_bookcorpus(source_text):
        return len(source_text.split()) > 50

    source_column = 'text'
    sources = []
    for source_text in examples[source_column]:
        if source_text is not None and filter_bookcorpus(source_text):
            sources.append(preprocess_bookcorpus(source_text))

    new_examples = {
        source_column: sources,
    }

    return new_examples
