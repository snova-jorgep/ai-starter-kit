import os
import json
import yaml
from snsdk import SnSdk
from typing import Optional, List
import subprocess
import logging
import re

SNAPI_PATH = "~/.snapi"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

from dotenv import load_dotenv

load_dotenv("../.env", override=True)


class SnsdkWrapper:
    """init"""

    def __init__(self, config_path=None) -> None:

        self.config = None
        self.tenant_name = None
        self.snapi_path = SNAPI_PATH

        if config_path is not None:
            self.config = self._load_config(config_path)
            config_snapi_path = self.config["sambastudio"]["snapi_path"]
            if config_snapi_path is not None and len(config_snapi_path) > 0:
                self.snapi_path = self.config["sambastudio"]["snapi_path"]

        host_url, tenant_id, access_key = self._get_sambastudio_variables()

        print(host_url)
        print(tenant_id)
        print(access_key)

        # self.snsdk_client = SnSdk(
        #     host_url=host_url,
        #     access_key=access_key,
        #     tenant_id=tenant_id,
        # )

    """Tenant methods"""

    def list_tenants(self, verbose=False) -> list | None:
        """Lists all tenants

        Returns:
            list | None: list of existing tenants. If there's an error, None is returned.
        """
        list_tenant_response = self.snsdk_client.list_tenants()
        if list_tenant_response["status_code"] == 200:
            tenants = []
            if verbose:
                return list_tenant_response["data"]
            else:
                for tenant in list_tenant_response["data"]:
                    tenants.append(
                        {
                            "tenant_id": tenant.get("tenant_id"),
                            "tenant_name": tenant.get("tenant_name"),
                        }
                    )
                return tenants
        else:
            logging.error(
                f"Failed to list projects. Details: {list_tenant_response['detail']}"
            )
            raise Exception(f"Error message: {list_tenant_response['detail']}")

    def search_tenant(self, tenant_name: str = None) -> str | None:
        """Search tenant

        Args:
            tenant_id (str): tenant name to search

        Returns:
            str: searched tenant information. If there's an error, None is returned.
        """
        if tenant_name is None:
            tenant_name = self.config["sambastudio"]["tenant_name"]

        tenant_info_response = self.snsdk_client.tenant_info(tenant=tenant_name)
        if tenant_info_response["status_code"] == 200:
            tenant_id = tenant_info_response["data"]["tenant_id"]
            logging.info(f"Tenant with name '{tenant_name}' found with id {tenant_id}")
            return tenant_id
        else:
            logging.info(f"Tenant with name '{tenant_name}' not found")
            return None

    """Internal methods"""

    def _get_sambastudio_variables(self) -> tuple:
        """Gets Sambastudio host name, tenant id and access key from Snapi folder location

        Raises:
            FileNotFoundError: raises error when the snapi config or secret file is not found
            ValueError: raises error when the snapi config file doesn't contain a correct json format

        Returns:
            tuple: host name, tenant id and access key from snapi setup
        """
        snapi_config = ""
        snapi_secret = ""

        try:

            # reads snapi config json
            snapi_config_path = os.path.expanduser(self.snapi_path) + "/config.json"
            with open(snapi_config_path, "r") as file:
                snapi_config = json.load(file)

            # reads snapi secret txt file
            snapi_secret_path = os.path.expanduser(self.snapi_path) + "/secret.txt"
            with open(snapi_secret_path, "r") as file:
                snapi_secret = file.read()

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: The file {snapi_config_path} does not exist."
            )
        except ValueError:
            raise ValueError(
                f"Error: The file {snapi_config_path} contains invalid JSON."
            )

        host_name = os.getenv("HOST_NAME")
        access_key = os.getenv("ACCESS_KEY")
        tenant_name = os.getenv("TENANT_NAME")

        if (
            (host_name is not None)
            and (access_key is not None)
            and (tenant_name is not None)
        ):
            new_snapi_config = {}
            new_snapi_config["HOST_NAME"] = host_name
            new_snapi_config["CONFIG_DIR"] = snapi_config["CONFIG_DIR"]
            new_snapi_config["DISABLE_SSL_WARNINGS"] = snapi_config[
                "DISABLE_SSL_WARNINGS"
            ]
            with open(snapi_config_path, "w") as file:
                json.dump(new_snapi_config, file)
            with open(snapi_secret_path, "w") as file:
                file.write(access_key)

            snapi_config_response = subprocess.run(
                ["snapi", "config", "set", "--tenant", "default"],
                capture_output=True,
                text=True,
            )

            with open(snapi_config_path, "r") as file:
                new_snapi_config = json.load(file)
            new_new_snapi_config = new_snapi_config

            tmp_snsdk_client = SnSdk(
                host_url=host_name,
                access_key=access_key,
                tenant_id=new_snapi_config["TENANT_ID"],
            )

            tenant_info_response = tmp_snsdk_client.tenant_info(
                tenant=new_snapi_config["TENANT_ID"]
            )
            tenant_id = tenant_info_response["data"]["tenant_id"]
            new_new_snapi_config["TENANT_ID"] = tenant_id

            with open(snapi_config_path, "w") as file:
                json.dump(new_new_snapi_config, file)

            return host_name, tenant_id, access_key

        else:

            host_name = snapi_config["HOST_NAME"]
            tenant_id = snapi_config["TENANT_ID"]
            access_key = snapi_secret

        return host_name, tenant_id, access_key

    def _load_config(self, file_path):
        """Loads a YAML configuration file.

        Args:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: The configuration data loaded from the YAML file.
        """
        try:
            with open(file_path, "r") as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file {file_path} does not exist.")
        except yaml.scanner.ScannerError:
            raise ValueError(f"Error: The file {file_path} contains invalid yaml.")
        return config

    """Dataset"""

    # jorge's implementation
    def list_datasets(self):
        list_models_response = self.snsdk_client.list_datasets()
        if list_models_response["status_code"] == 200:
            models = {}
            for model in list_models_response["models"]:
                if set(["train", "deploy"]).issubset(model.get("jobTypes")):
                    models[model.get("model_checkpoint_name")] = model.get("model_id")
            return models
        else:
            logging.error(
                f"Failed to list models. Details: {list_models_response['detail']}"
            )
            raise Exception(f"Error message: {list_models_response['detail']}")

    # jorge's implementation
    def search_dataset(self, dataset_name):
        search_dataset_response = self.snsdk_client.search_dataset(
            dataset_name=dataset_name
        )
        if search_dataset_response["status_code"] == 200:
            dataset_id = search_dataset_response["data"]["dataset_id"]
            logging.info(
                f"Dataset with name '{dataset_name}' found with id {dataset_id}"
            )
            return dataset_id
        else:
            logging.info(f"Dataset with name '{dataset_name}' not found")
            return None

    def delete_dataset(self, dataset):
        if dataset is not None:
            dataset = self.config["project"]["project_name"]
        delete_dataset_response = self.snsdk_client.delete_dataset(dataset=dataset)
        if delete_dataset_response["status_code"] == 200:
            logging.info(f"Dataset with name or id '{dataset}' deleted")
        else:
            logging.error(
                f"Failed to delete project with name or id '{dataset}'. Details: {delete_dataset_response}"
            )
            raise Exception(f"Error message: {delete_dataset_response}")

    def create_dataset(
        self,
        dataset_name: Optional[str] = None,
        dataset_description: Optional[str] = None,
        job_type: Optional[str] = None,
        dataset_app_names: Optional[List[str]] = None,
        dataset_source_type: Optional[str] = None,
        dataset_source_file: Optional[str] = None,
        dataset_language: Optional[str] = None,
    ):

        # check if apps selected exist
        if dataset_app_names is None:
            dataset_app_names = self.config["dataset"]["dataset_app_names"]
        for app_name in dataset_app_names:
            app_id = self.search_app(app_name)
            if app_id is None:
                raise Exception(f"App '{app_name}' not found")

        # check if dataset already exists
        dataset_name = dataset_name or self.config.get("dataset", {}).get(
            "dataset_name"
        )
        dataset_id = self.search_dataset(dataset_name)

        if dataset_id is None:
            command = [
                "snapi",
                "dataset",
                "add",
                "--dataset-name",
                dataset_name,
                "--description",
                dataset_description
                or self.config.get("dataset", {}).get("dataset_description"),
                "--job_type",
                job_type or self.config.get("job", {}).get("job_type"),
                "--apps",
                # TODO: test with list of apps
                dataset_app_names[0],
                "--source_type",
                dataset_source_type
                or self.config.get("dataset", {}).get("dataset_source_type"),
                "--source_file",
                dataset_source_file
                or self.config.get("dataset", {}).get("dataset_source_file"),
                "--language",
                dataset_language
                or self.config.get("dataset", {}).get("dataset_language"),
            ]
            echo_response = subprocess.run(
                ["echo", "yes"], capture_output=True, text=True
            )
            snapi_response = subprocess.run(
                command, input=echo_response.stdout, capture_output=True, text=True
            )
            # print(snapi_response)
            if (
                ("status_code" in snapi_response.stdout.lower())
                and ("error occured" in snapi_response.stdout.lower())
                or (len(snapi_response.stderr) > 0)
            ):
                if len(snapi_response.stderr) > 0:
                    error_message = snapi_response.stderr
                else:
                    error_message = re.search(
                        r"message:\s*(.*)", snapi_response.stdout
                    )[0]
                logging.error(
                    f"Failed to create dataset with name '{dataset_name}'. Details: {error_message}"
                )
                raise Exception(f"Error message: {error_message}")
            else:
                dataset_id = self.search_dataset(dataset_name=dataset_name)
                logging.info(
                    f"Dataset with name '{dataset_name}' created: '{snapi_response.stdout}'"
                )
                return dataset_id
        else:
            logging.info(
                f"Project with name '{dataset_name}' already exists with id '{dataset_id}', using it"
            )
        return dataset_id

    def check_dataset_creation_progress(self, dataset_name):
        # check progress in case data is huge
        # e2e process has to wait while this is working, showing progress
        pass

    """generic stuff"""

    def list_models():
        pass

    def list_projects():
        pass

    """app"""

    def list_apps(self, verbose=False):
        response = self.snsdk_client.list_apps()
        if response["status_code"] == 200:
            apps = []
            if verbose:
                apps = response["apps"]
            else:
                for app in response["apps"]:
                    apps.append({"id": app.get("id"), "name": app.get("name")})
            return apps
        else:
            logging.error(f"Failed to list models. Details: {response['detail']}")
            raise Exception(f"Error message: {response['detail']}")

    def list_apps(self, verbose=False):
        response = self.snsdk_client.list_apps()
        if response["status_code"] == 200:
            apps = []
            if verbose:
                apps = response["apps"]
            else:
                for app in response["apps"]:
                    apps.append({"id": app.get("id"), "name": app.get("name")})
            return apps
        else:
            logging.error(f"Failed to list apps. Details: {response['detail']}")
            raise Exception(f"Error message: {response['detail']}")

    def search_app(self, app_name):
        app_id = None
        try:
            app_info_response = self.snsdk_client.app_info(app=app_name)["apps"]
            app_id = app_info_response["id"]
        except Exception as e:
            logging.error(
                f"Failed to retrieve information for app. Details: {app_name}"
            )
            raise Exception(f"Error message: {e}")
        return app_id


if __name__ == "__main__":

    config_path = "./fine_tuning/config.yaml"
    snsdkwrapper_client = SnsdkWrapper(config_path=config_path)

    # app_name = snsdkwrapper_client.list_apps()[0].get("name")
    # app_id = snsdkwrapper_client.search_app(app_name=app_name)

    # print(snsdkwrapper_client.list_apps()[0])
    # print(app_name)
    # print(app_id)
