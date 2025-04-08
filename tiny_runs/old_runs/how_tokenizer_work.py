from transformers import AutoTokenizer


def read_data():
    return [
        "Apollo 11 was the first manned space mission to land humans on the Moon. "
        "Led by Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin, the American lunar crew landed aboard the lunar module \"Eagle\" on the moon's surface at 20:17 UTC on July 20, 1969. Six hours and 39 minutes later, on July 21 at 02:56 UTC, Armstrong became the first person to set foot on the lunar surface. "
        "Aldrin followed 19 minutes later. The two spent approximately 2 hours and 15 minutes (135 minutes) on the lunar surface, collecting 21.55 kilograms of lunar rock samples to bring back to Earth. While Armstrong and Aldrin were conducting their activities on the Moon, Command Module Pilot Michael Collins was alone orbiting the Moon in the service and command module \"Columbia.\" The commander and the lunar module pilot spent 21 hours and 36 minutes on the lunar surface, naming the landing site Tranquility Base before they lifted off in the ascent stage of the lunar module and rendezvoused with \"Columbia.\" On July 16 at 13:32 UTC, the Saturn V rocket carrying Apollo 11 lifted off from Kennedy Space Center at Merritt Island, Florida. This was the fifth manned space mission in NASA's Apollo program. The Apollo spacecraft consisted of three sections: only the command module, capable of housing the three astronauts, would return to Earth. The service module provided propulsion, power, oxygen, and water for the command module. The lunar module came in two parts, with the descent stage for landing on the moon's surface, and the ascent stage to lift the astronauts back into lunar orbit. The third stage of the Saturn V rocket placed the spacecraft into a trans-lunar trajectory. The astronauts separated the spacecraft from the rocket and flew for three days before entering lunar orbit. Armstrong and Aldrin entered the lunar module and landed in the Sea of Tranquility on July 20, completing the lunar surface tasks before lifting off from \"Eagle's\" ascent stage to rendezvous with Collins in the command module. The trio subsequently jettisoned \"Eagle,\" propelling \"Columbia\" out of lunar orbit and into a return trajectory toward Earth. On July 24, after a mission lasting over eight days, they splashed down in the Pacific Ocean."

        "Questions (Answer Questions One by One): 1. When did the first manned mission to the moon take place? 2. Who were the crew members of the Apollo 11 mission? 3. Who was the second person to step on the moon?\n"
        "Answer: "
    ]


if __name__ == '__main__':
    tok = AutoTokenizer.from_pretrained("/input/jitai/huggingface/hub/Lourdle/Llama-3-8B-Instruct-262k")
    data = read_data()[0]
    res = tok.encode(data)
    id_vocab = {v:k for k,v in tok.vocab.items()}
    res = [id_vocab[i] for i in res]
    print(res)
    res = '-'.join(res).replace('Ġ', '')
    print(res)
