.PHONY: setup quick http package clean

setup:
	python -m pip install -r requirements.txt

quick:
	bash scripts/run_quick_benchmark.sh

http:
	bash scripts/run_http_smoke.sh

package:
	bash scripts/make_submission_zip.sh

clean:
	rm -rf benchmark/results dist __pycache__ benchmark/__pycache__ .pytest_cache
